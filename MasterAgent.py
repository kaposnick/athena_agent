import multiprocessing as mp

from attr import has
from BaseAgent import BaseAgent
from Buffer import EpisodeBuffer

from common_utils import MODE_INFERENCE, MODE_SCHEDULING_AC, MODE_SCHEDULING_RANDOM, MODE_TRAINING, get_basic_actor_network, get_basic_critic_network, get_shared_memory_ref, import_tensorflow, map_weights_to_shared_memory_buffer, publish_weights_to_shared_memory, save_weights
import random
import numpy as np

class Master_Agent(mp.Process):
    def __init__(self,
                environment,
                hyperparameters,
                config, state_size, action_size,
                actor_memory_bytes: mp.Value, critic_memory_bytes: mp.Value,
                memory_created: mp.Value, master_agent_initialized: mp.Value,
                sample_buffer_queue: mp.Queue, batch_info_queue: mp.Value,
                results_queue: mp.Queue,
                master_agent_stop: mp.Value,
                optimizer_lock: mp.Lock,
                in_training_mode: mp.Value,
                scheduling_mode,
                actor_memory_name = 'model_actor',
                critic_memory_name = 'model_critic') -> None:
        super(Master_Agent, self).__init__()
        self.environment = environment
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.hyperparameters = hyperparameters
        self.actor_memory_bytes = actor_memory_bytes
        self.critic_memory_bytes = critic_memory_bytes
        self.memory_created = memory_created
        self.master_agent_initialized = master_agent_initialized
        self.sample_buffer_queue = sample_buffer_queue
        self.batch_info_queue    = batch_info_queue
        self.results_queue = results_queue
        self.optimizer_lock = optimizer_lock
        self.master_agent_stop = master_agent_stop
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']
        self.in_training_mode = in_training_mode
        self.scheduling_mode  = scheduling_mode

    def __str__(self) -> str:
        return 'Master Agent'

    def set_process_seeds(self):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        if (hasattr(self, 'tf')):
            self.tf.random.set_seed(self.config.seed)

    def compute_model_size(self, model):
        model_dtype = np.dtype(model.dtype)
        variables = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        size = variables * model_dtype.itemsize
        return size, model_dtype

    def get_shared_memory_reference(self, model, memory_name):
        ## return the reference to the memory and the np.array pointing to the shared memory
        size, model_dtype = self.compute_model_size(model)
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initiate_models(self):
        try:
            stage = 'Creating the neural networks'
            models = [
                get_basic_actor_network(self.tf, self.tfp, self.state_size), 
                get_basic_critic_network(self.tf, self.state_size, 1)
            ]

            self.optimizer = self.tf.keras.optimizers.Adam(
                learning_rate = self.hyperparameters['Actor_Critic_Common']['learning_rate']                
            )

            self.actor = models[0]
            self.actor_memory_bytes.value, actor_dtype = self.compute_model_size(self.actor)
            
            self.critic = models[1]
            self.critic_memory_bytes.value, critic_dtype = self.compute_model_size(self.critic)
            while (self.memory_created.value == 0):
                pass

            stage = 'Actor memory reference creation'
            self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.actor, self.actor_memory_name)
            self.weights_actor = map_weights_to_shared_memory_buffer(self.actor.get_weights(), self.np_array_actor)

            stage = 'Critic memory reference creation'
            self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.critic, self.critic_memory_name)
            self.weights_critic = map_weights_to_shared_memory_buffer(self.critic.get_weights(), self.np_array_critic)
        except Exception as e:
            print(str(self) + ' -> Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e
    
    def initiate_variables(self):
        self.buffer = EpisodeBuffer(100000, self.batch_size, self.state_size, 1)
        self.include_entropy_term = self.config.hyperparameters['Actor_Critic_Common']['include_entropy_term']
        self.entropy_contribution = self.config.hyperparameters['Actor_Critic_Common']['entropy_contribution']

    def fn_return_first(self, distr, tensor):
        tf = self.tf
        exact = tf.cast(tensor.lookup(tf.constant(0)), dtype=tf.float32)
        higher = tf.cast(tensor.lookup(tf.constant(1)), dtype=tf.float32)
        higher_half = tf.math.divide(tf.add(exact, higher), tf.constant(2.0))
        
        distr1 = distr.cdf(higher_half)
        distr2 = distr.cdf(exact)

        result = tf.math.subtract(distr1, distr2)
        return result
        # return tf.expand_dims(tf.math.subtract(distr1, distr2), axis = 1)

    def fn_return_last(self, distr, tensor):
        tf = self.tf        
        exact = tf.cast(tensor.lookup(tf.constant(self.tbs_len-1)), dtype = tf.float32)
        lower = tf.cast(tensor.lookup(tf.constant(self.tbs_len-2)), dtype = tf.float32)
        lower_half = tf.math.divide(tf.add(exact, lower), tf.constant(2.0))
        
        distr1 = distr.cdf(exact)
        distr2 = distr.cdf(lower_half)
        
        result = tf.math.subtract(distr1, distr2)
        return result
        # return tf.expand_dims(tf.math.subtract(distr1, distr2), axis = 1)

    def fn_return_medio(self, distr, tensor, idx):
        tf = self.tf
        exact = tf.cast(tensor.lookup(idx), dtype=tf.float32)
        higher = tf.cast(tensor.lookup(idx + 1), dtype=tf.float32)
        lower = tf.cast(tensor.lookup(idx - 1), dtype=tf.float32)
        higher_half = tf.math.divide(tf.add(exact, higher), tf.constant(2.0))
        lower_half  = tf.math.divide(tf.add(exact, lower), tf.constant(2.0))

        distr_1 = distr.cdf(higher_half)
        distr_2 = distr.cdf(lower_half)
        # result = tf.expand_dims(tf.subtract(distr_1, distr_2), axis = 1)
        result = tf.subtract(distr_1, distr_2)
        return result

    def fn_prob(self, distr, action):
        tf = self.tf
        idx = self.tbs_values_to_tbs_idx_tensor.lookup(tf.cast(action, dtype=tf.int32))
        r = tf.where(
            tf.equal(idx, 0),
            self.fn_return_first(distr, self.tbs_idx_to_tbs_values_tensor),
            tf.where(
                tf.equal(idx, self.tbs_len - 1),
                self.fn_return_last(distr, self.tbs_idx_to_tbs_values_tensor),
                self.fn_return_medio(distr, self.tbs_idx_to_tbs_values_tensor, idx)
            )
        )
        return r

    def fn_log_prob(self,distr, action):
        tf = self.tf
        return tf.math.log(self.fn_prob(distr, action) + 1e-7)

    def critic_learn(self, state, reward, record_info = False):
        tf = self.tf
        with tf.GradientTape() as tape:
            td = self.critic(state, training = True) - reward
            loss = tf.math.reduce_mean(tf.math.square(td))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        grads = [(tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        info = {}
        if (record_info):
            info['critic_loss'] = self.tf.math.reduce_mean(loss)
        return info

    def actor_learn(self, state, action, reward, critic, add_entropy_term, gradients_update_idx, record_info = False):
        tf = self.tf
        info = {}
        with tf.GradientTape() as tape:
            mu, sigma    =  tf.unstack(self.actor(state, training = True), num = 2, axis = -1)
            mu = tf.expand_dims(mu, axis = 1)
            sigma = 1e-5 + tf.math.softplus(tf.expand_dims(sigma, axis = 1))
            advantage = reward - critic

            distr_batch = self.tfp.distributions.Normal(loc = mu, scale = sigma)            
            nll = self.fn_log_prob(distr_batch, action)
            actor_advantage_nll = nll * advantage
            actor_loss = actor_advantage_nll
            if (add_entropy_term and self.include_entropy_term):
                entropy = distr_batch.entropy()
                if (record_info):
                    info['entropy']  = self.tf.math.reduce_mean(entropy)
                constant = tf.constant(.995)
                actor_loss += self.entropy_contribution * entropy * tf.math.pow(constant, gradients_update_idx)
                        
            actor_loss  = -1 * self.tf.math.reduce_mean(actor_loss)
        grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        grads = [(tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        if (record_info):
            info['actor_loss'] = self.tf.math.reduce_mean(actor_loss)
            info['reward'] = self.tf.math.reduce_mean(reward)
            info['actor_nll'] = self.tf.math.reduce_mean(actor_advantage_nll)
            info['action_mean'] = self.tf.math.reduce_mean(distr_batch.mean())
            info['action_stddev'] = self.tf.math.reduce_mean(distr_batch.stddev())
        return info


    def learn(self, gradients_update_idx, epochs = 3, sil_updates = 3):
        # This function performs the policy iteration algorithm which comprises 
        # the policy evaluation and policy improvement.
        # First it calculates the A2C updates and later the SIL updates.
        # The policy evaluation consists of minimizing the MSE of the critic network.
        # The policy improvement consists of maximizing the NLL of the actor network.
        
        print(str(self) + '[{}] -> Learning...'.format(gradients_update_idx))
        info = {}

        # A2C Learning
        a2c_state, a2c_action, a2c_reward = self.buffer.sample(self.tf)
        print('A2C Critic Loss: ', end = '')
        for i in range(epochs):
            critic_info = self.tf_critic_learn(a2c_state, a2c_reward, record_info = True)
            print('{:.2f}, '.format(critic_info['critic_loss'].numpy()), end = '')
        print('Done')

        critic_reward = self.critic(a2c_state)
        print('A2C Actor Loss: ', end = '')
        for i in range(epochs):
            actor_info = self.tf_actor_learn(
                a2c_state, a2c_action, a2c_reward, critic_reward, 
                add_entropy_term = True, gradients_update_idx = gradients_update_idx, record_info = True)
            print('{:.2f}, '.format(actor_info['actor_nll'].numpy()), end = '')
        print('Done')

        # SIL
        for _ in range(sil_updates):
            sil_state, sil_action, sil_reward = self.buffer.sample_sil(self.tf, self.critic, uniform = False)
            for _ in range(epochs):
                self.tf_critic_learn(sil_state, sil_reward, record_info = False )

            critic_reward = self.critic(sil_state)
            for _ in range(epochs):
                self.tf_actor_learn(sil_state, sil_action, sil_reward, critic_reward, add_entropy_term = False, gradients_update_idx = None, record_info = False)
        
        info = {
            'actor_loss' : actor_info['actor_loss'].numpy(),
            'critic_loss': critic_info['critic_loss'].numpy(),
            'reward'     : actor_info['reward'].numpy(),
            'entropy'    : actor_info['entropy'].numpy(),
            'actor_nll'  : actor_info['actor_nll'].numpy(),
            'action_mean': actor_info['action_mean'].numpy(),
            'action_stdv': actor_info['action_stddev'].numpy()
        }
        return info

    def save_weights(self, suffix = ''):
        if (self.config.save_weights):
            save_weights(self.actor, self.config.save_weights_file + suffix + '_actor.h5', False)
            save_weights(self.critic, self.config.save_weights_file + suffix + '_critic.h5')

    def apply_gradients(self, actor_grads, critic_grads):
        actor_grads = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in actor_grads]
        self.optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_weights)
        )
        critic_grads = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in critic_grads]
        self.optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_weights)
        )

    def send_results(self):
        mcs_mean, prb_mean = self.environment.calculate_mean(None, self.batch_info)
        additional_columns = self.environment.get_csv_result_policy_output(self.batch_info)

        self.results_queue.put([ [mcs_mean, prb_mean], additional_columns ])

    def create_array(self):
        tf = self.tf
        tbs_array = np.array(self.environment.get_tbs_array(), dtype=np.float32)
        tbs_map = {}
        idx = 0
        for tbs_value in tbs_array:
            tbs_map[tbs_value] = idx
            idx += 1
        tbs_values = tf.cast(tf.constant(list(tbs_map.keys())), dtype=tf.int32)
        tbs_idx = tf.cast(tf.constant(list(tbs_map.values())), dtype = tf.int32)
        self.tbs_idx_to_tbs_values_tensor = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tbs_idx, tbs_values),
            default_value = 1992
        )

        self.tbs_values_to_tbs_idx_tensor = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tbs_values, tbs_idx),
            default_value = 65
        )
        self.tbs_len = len(tbs_array)

    def run_in_collecting_stats_mode(self):
        self.set_process_seeds()
        sample_idx = 0
        self.batch_info = []
        try:
            self.master_agent_initialized.value = 1
            while True:
                import queue
                try:
                    if (self.master_agent_stop.value == 1):
                        break
                    while (sample_idx < 1):
                        info_list = self.batch_info_queue.get()
                        for info in info_list:
                            self.batch_info.append(info)
                        sample_idx += 1
                    self.send_results()
                    sample_idx = 0
                    self.batch_info = []
                except queue.Empty:
                    if (self.master_agent_stop.value == 1):
                        exited_successfully = True
                        break
            pass
        finally:
            print(str(self) + ' -> Exiting ...')


    def exeucte_in_schedule_random_mode(self):
        self.set_process_seeds()
        self.master_agent_initialized.value = 1
        import queue
        filename = self.config.results_file_path
        with open(filename, 'w') as file:
            file.write('|'.join(['cpu', 'snr', 'mcs', 'prb', 'crc', 'decoding_time']) + '\n')
            sample_idx = 0
            while True:
                try:
                    if (self.master_agent_stop.value == 1):
                        break
                    sample_buffer = self.sample_buffer_queue.get(block = True, timeout = 10)
                    for sample in sample_buffer:
                        state = sample[0] # snr, cpu
                        action = sample[1] # mcs, prb
                        reward = sample[2] # crc, time
                        
                        record = [str(state[1]), str(state[0]), str(action[0]), str(action[1]), str(reward[0]), str(reward[1])]
                        file.write('|'.join(record) + '\n')
                    sample_idx += 1
                    if (sample_idx == 10):
                        file.flush()
                        sample_idx = 0
                except queue.Empty:
                    pass


    def send_results_thread(self):
        while(True):
            self.batch_info = []
            info_list = self.batch_info_queue.get()
            for info in info_list:
                self.batch_info.append(info)
            self.send_results()
            self.batch_info = []

    def load_initial_weights_if_configured(self):
         if (self.config.load_initial_weights):
                print(str(self) + ' -> Loading actor initial weights from ' + self.config.initial_weights_path)
                self.actor.load_weights(self.config.initial_weights_path)
                print(str(self) + ' -> Loading critic initial weights from ' + self.config.critic_initial_weights_path)
                self.critic.load_weights(self.config.critic_initial_weights_path)

    def publish_weights(self):
        publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)                
        print(str(self) + ' -> Published actor weights to shared memory')
        
        publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)
        print(str(self) + ' -> Published critic weights to shared memory')

    def execute_in_schedule_ac_mode(self):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        import threading
        self.set_process_seeds()  
        exited_successfully = False
        try:
            self.initiate_models()
            self.initiate_variables()
            self.create_array()
            self.tf_actor_learn = self.tf.function(self.actor_learn)
            self.tf_critic_learn = self.tf.function(self.critic_learn)

            self.load_initial_weights_if_configured()           
            self.publish_weights()
            
            threading.Thread(target=self.send_results_thread).start()
            self.master_agent_initialized.value = 1

            gradient_calculation_idx = 0
            sample_idx = 0
            self.buffer.reset_episode()
            has_entered_inference_mode = False
            while True:
                import queue
                try:
                    if (self.master_agent_stop.value == 1):
                        break
                    if (not has_entered_inference_mode) and (self.in_training_mode.value == MODE_INFERENCE):
                        has_entered_inference_mode = True
                        print(str(self) + ' -> Entering inference mode...')
                    while (sample_idx < self.batch_size):
                        if (self.in_training_mode.value == MODE_TRAINING):
                            record_list = self.sample_buffer_queue.get(block = True, timeout = 10)
                            for record in record_list:
                                self.buffer.record(record)
                        elif (self.in_training_mode.value == MODE_INFERENCE):
                            break
                        sample_idx += 1

                    if (self.in_training_mode.value == MODE_TRAINING):
                        self.learn(gradient_calculation_idx)
                    
                    sample_idx = 0
                    self.buffer.reset_episode()
                    

                    if (self.in_training_mode.value == MODE_TRAINING):
                        with self.optimizer_lock:
                            # print(str(self) + ' -> Pushing new weights...')
                            publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)
                            publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)
                        gradient_calculation_idx += 1
                        if (gradient_calculation_idx % self.config.save_weights_period == 0):
                            self.save_weights('_training_')

                except queue.Empty:
                    if (self.master_agent_stop.value == 1):
                        exited_successfully = True
                        break
        finally:
            print(str(self) + ' -> Exiting ...')
            self.save_weights('_exiting_')

    def run(self) -> None:
        if (self.scheduling_mode == MODE_SCHEDULING_AC):
            self.execute_in_schedule_ac_mode()
        elif (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            self.exeucte_in_schedule_random_mode()
        else: 
            self.run_in_collecting_stats_mode()

        
        


