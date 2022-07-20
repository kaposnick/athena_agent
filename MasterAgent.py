import multiprocessing as mp
from BaseAgent import BaseAgent
from Buffer import EpisodeBuffer

from common_utils import get_basic_actor_network, get_basic_critic_network, get_shared_memory_ref, import_tensorflow, map_weights_to_shared_memory_buffer, publish_weights_to_shared_memory, save_weights
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
                episode_number: mp.Value,
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
        self.episode_number = episode_number
        self.master_agent_stop = master_agent_stop
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']

    def __str__(self) -> str:
        return 'Master Agent'

    def set_process_seeds(self):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
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
        self.buffer = EpisodeBuffer(20000, self.batch_size, self.state_size, 1)
        self.include_entropy_term = self.config.hyperparameters['Actor_Critic_Common']['include_entropy_term']
        self.entropy_contribution = self.config.hyperparameters['Actor_Critic_Common']['entropy_contribution']

    def compute_grads(self, state_batch, action_batch, reward_batch, add_entropy_term = True):            
        info = {}
        with self.tf.GradientTape() as tape1, self.tf.GradientTape() as tape2:
            distr_batch    =  self.actor(state_batch, training = True)
            critic_batch   = self.critic(state_batch, training = True)
            advantage = reward_batch - critic_batch
            
            nll = distr_batch.log_prob(action_batch)
            actor_advantage_nll = nll * advantage
            actor_loss = actor_advantage_nll

            if (add_entropy_term and self.include_entropy_term):
                entropy = distr_batch.entropy()
                info['entropy']  = self.tf.math.reduce_mean(entropy)
                actor_loss -= self.entropy_contribution * entropy
                        
            actor_loss  = -1 * self.tf.math.reduce_mean(actor_loss)
            critic_loss = 0.5 * self.tf.math.reduce_mean(self.tf.math.square(advantage))
            info['actor_loss'] = self.tf.math.reduce_mean(actor_loss)
            info['critic_loss'] = self.tf.math.reduce_mean(critic_loss)
            info['reward'] = self.tf.math.reduce_mean(reward_batch)
            info['actor_nll'] = self.tf.math.reduce_mean(actor_advantage_nll)
            info['action_mean'] = self.tf.math.reduce_mean(distr_batch.mean())
            info['action_stddev'] = self.tf.math.reduce_mean(distr_batch.stddev())
        
        actor_grads  = tape1.gradient(actor_loss, self.actor.trainable_weights)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_weights)

        return actor_grads, critic_grads, info

    def compute_sil_grads(self):
        state_batch, action_batch, reward_batch = self.buffer.sample_sil(self.tf, self.critic)
        return self.tf_compute_grads(state_batch, action_batch, reward_batch, add_entropy_term = False)

    def compute_a2c_grads(self):
        state_batch, action_batch, reward_batch = self.buffer.sample(self.tf)
        return self.tf_compute_grads(state_batch, action_batch, reward_batch, add_entropy_term = True)


    def learn(self):
        a2c_actor_grads, a2c_critic_grads, a2c_info = self.compute_a2c_grads()
        sil_actor_grads, sil_critic_grads, sil_info = self.compute_sil_grads()
        self.tf_apply_gradients(a2c_actor_grads, a2c_critic_grads, sil_actor_grads, sil_critic_grads)
        info = {
            'actor_loss' : a2c_info['actor_loss'].numpy(),
            'critic_loss': a2c_info['critic_loss'].numpy(),
            'reward'     : a2c_info['reward'].numpy(),
            'entropy'    : a2c_info['entropy'].numpy(),
            'actor_nll'  : a2c_info['actor_nll'].numpy(),
            'action_mean': a2c_info['action_mean'].numpy(),
            'action_stdv': a2c_info['action_stddev'].numpy(),
            'sil_actor_loss' : sil_info['actor_loss'].numpy(),
            'sil_critic_loss': sil_info['critic_loss'].numpy(),
            'sil_reward'     : sil_info['reward'].numpy(),
            'sil_actor_nll'  : sil_info['actor_nll'].numpy(),
            'sil_action_mean': sil_info['action_mean'].numpy(),
            'sil_action_stdv': sil_info['action_stddev'].numpy()
        }
        return info

    def save_weights(self):
        save_weights(self.actor, self.config.save_weights_file, False)
        save_weights(self.critic, self.config.save_weights_file + '_critic.h5')

    def apply_gradients(self, a2c_ac_grads, a2c_cr_grads, sil_ac_grads, sil_cr_grads):
        actor_gradients = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in a2c_ac_grads]
        self.actor_critic_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_weights)
        )
        critic_gradients = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in a2c_cr_grads]
        self.actor_critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_weights)
        )
        
        if (sil_ac_grads is not None and sil_cr_grads is not None):
            actor_gradients = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1))  for grad in sil_ac_grads]
            self.actor_critic_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_weights))
            critic_gradients = [(self.tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in sil_cr_grads]
            self.actor_critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_weights))

    def send_results(self, info):
        if (True):
            actor_loss     = info['actor_loss']
            critic_loss    = info['critic_loss']
            rew_mean       = info['reward']
            entropy        = info['entropy']
            actor_nll      = info['actor_nll']
            tbs_mean       = info['action_mean']
            tbs_stdv       = info['action_stdv']
            mcs_mean, prb_mean = self.environment.calculate_mean(None, self.batch_info)
        else:
            actor_loss = '-1'
            critic_loss = '-1'
            rew_mean = '-1'
            entropy   = '-1'
            actor_nll = '-1'
            mcs_mean = '-1'
            prb_mean = '-1'
            tbs_mean = '-1'
            tbs_stdv = '-1'
            mcs_mean, prb_mean = self.environment.calculate_mean(None, self.batch_info)
        additional_columns = self.environment.get_csv_result_policy_output(self.batch_info)

        self.results_queue.put([ [rew_mean, actor_nll, entropy, mcs_mean, prb_mean, tbs_mean, tbs_stdv, actor_loss, critic_loss], additional_columns ])

    def run(self):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds()  
        exited_successfully = False
        try:
            self.initiate_models()
            self.initiate_variables()       
            self.tf_apply_gradients = self.tf.function(self.apply_gradients)
            self.tf_compute_grads   = self.tf.function(self.compute_grads)

            if (self.config.load_initial_weights):
                print(str(self) + ' -> Loading initial weights from ' + self.config.initial_weights_path)
                self.actor.load_weights(self.config.initial_weights_path)
            
            publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)                
            print(str(self) + ' -> Published actor weights to shared memory')
            
            publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)
            print(str(self) + ' -> Published critic weights to shared memory')
            self.master_agent_initialized.value = 1

            lr_schedule = self.tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = self.hyperparameters['Actor_Critic_Common']['learning_rate'],
                decay_steps = 3,
                decay_rate  = 0.995
            )
            self.actor_critic_optimizer = self.tf.keras.optimizers.Adam(learning_rate = lr_schedule)
            gradient_calculation_idx = 0
            sample_idx = 0
            self.buffer.reset_episode()
            self.batch_info = []
            while True:
                import queue
                try:
                    while (sample_idx < 1):
                        record_list = self.sample_buffer_queue.get(block = True, timeout = 10)
                        for record in record_list:
                            self.buffer.record(record)
                        info_list = self.batch_info_queue.get()
                        for info in info_list:
                            self.batch_info.append(info)
                        sample_idx += 1

                    info = self.learn()
                    self.send_results(info)
                    
                    sample_idx = 0
                    self.buffer.reset_episode()
                    self.batch_info = []

                    with self.optimizer_lock:
                        print(str(self) + ' -> Pushing new weights...')
                        publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)
                        publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)

                    with self.episode_number.get_lock():
                        self.episode_number.value += 1

                    gradient_calculation_idx += 1
                    if (self.config.save_weights and gradient_calculation_idx % self.config.save_weights_period == 0):
                        self.save_weights()
                except queue.Empty:
                    if (self.master_agent_stop.value == 1):
                        exited_successfully = True
                        break
        finally:
            print(str(self) + ' -> Exiting gracefully...')
            if (exited_successfully):
                if (self.config.save_weights):
                    self.save_weights()
        


