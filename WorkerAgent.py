import multiprocessing as mp
import numpy as np
import random
from OUActionNoise import OUActionNoise
from Buffer import Buffer
from env.BaseEnv import BaseEnv
from common_utils import get_basic_actor_network, get_basic_critic_network, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer
from Config import Config

class Actor_Critic_Worker(mp.Process):
    def __init__(self, 
                environment: BaseEnv, config: Config,
                worker_num: np.int32, total_workers: np.int32, 
                episodes_to_run: np.int32, state_size, action_size, 
                successfully_started_worker: mp.Value, 
                optimizer_lock: mp.Lock, write_to_results_queue_lock: mp.Lock,
                results_queue: mp.Queue, gradient_updates_queue: mp.Queue,
                episode_number: mp.Value,
                in_scheduling_mode: str,
                actor_memory_name = 'model_actor',
                critic_memory_name = 'model_critic') -> None:
        super(Actor_Critic_Worker, self).__init__()
        # environment variables
        self.environment = environment
        self.config = config
        self.worker_num = worker_num
        self.total_workers = total_workers
        self.episodes_to_run = episodes_to_run
        self.state_size = state_size
        self.action_size = action_size

        # multiprocessing variables
        self.successfully_started_worker = successfully_started_worker
        self.optimizer_lock = optimizer_lock
        self.write_to_results_queue_lock = write_to_results_queue_lock
        self.results_queue = results_queue
        self.gradient_updates_queue = gradient_updates_queue
        self.episode_number = episode_number
        self.in_scheduling_mode = in_scheduling_mode
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name

        self.init_configuration()        

    def __str__(self) -> str:
        return 'Worker ' + str(self.worker_num)
        
    def init_configuration(self):
        self.local_update_period = self.config.hyperparameters['Actor_Critic_Common']['local_update_period']
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']
        self.use_action_value_critic = not self.config.hyperparameters['Actor_Critic_Common']['use_state_value_critic']

    

    def set_process_seeds(self, tf, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed + worker_num)
        random.seed(self.config.seed + worker_num)
        np.random.seed(self.config.seed + worker_num)
        tf.random.set_seed(self.config.seed + worker_num)

    def get_shared_memory_reference(self, model, memory_name):
        ## return the reference to the memory and the np.array pointing to the shared memory
        model_dtype = np.dtype(model.dtype)
        variables = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        size = variables * model_dtype.itemsize
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initiate_models(self, tf):
        try:
            stage = 'Creating the neural networks'
            
            models = [
                get_basic_actor_network(tf, self.state_size), 
                get_basic_critic_network(tf, self.state_size, 1, self.action_size)
            ]

            stage = 'Actor memory reference creation'
            self.actor = models[0]
            self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.actor, self.actor_memory_name)
            self.weights_actor = map_weights_to_shared_memory_buffer(self.actor.get_weights(), self.np_array_actor)

            if (self.use_action_value_critic):
                stage = 'Action-value critic memory reference creation'
                self.critic = models[1]
                self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.critic, self.critic_memory_name)
                self.weights_critic = map_weights_to_shared_memory_buffer(self.critic.get_weights(), self.np_array_critic)
        except Exception as e:
            print(str(self) + ' -> Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def initiate_worker_variables(self, tf):
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.k = 10
        self.buffer = Buffer(self.batch_size, self.batch_size, self.state_size, 1)

    def update_weights(self):
        import copy
        with self.optimizer_lock:
            self.actor.set_weights(copy.deepcopy(self.weights_actor))
            if (self.use_action_value_critic):
                self.critic.set_weights(copy.deepcopy(self.weights_critic))

    def compute_critic_mse(self, tf):
        return tf.reduce_mean(self.advantage * self.advantage), None

    def compute_critic_loss_alt(self, tf):
        probs = []
        def negloglik(y_distr, y_true):
            neg_log_prob = -y_distr.log_prob(y_true)
            prob = tf.math.exp( -1 * neg_log_prob)
            probs.append(prob)
            return neg_log_prob
        critic_loss = tf.reduce_mean([negloglik(y_distr, y_true) for y_distr, y_true in zip(self.batch_critic_outputs, self.rewards)])
        return critic_loss, {'probs': tf.convert_to_tensor(np.mean(probs))}

    def compute_critic_loss(self, tf):
        critic_loss = tf.math.reduce_mean(tf.math.square(self.rewards, self.critic_mean))
        return critic_loss, None        

    def compute_actor_loss(self, tf):
        actor_loss = -1 * tf.math.reduce_mean(self.critic_mean)        
        return actor_loss, None

    def compute_losses(self, tf):
        self.rewards = tf.expand_dims(tf.convert_to_tensor(self.batch_rewards, dtype = tf.float32), axis = 1) # (batch_size, 1)
        self.actions = tf.convert_to_tensor(self.batch_action)
        self.critic_mean = tf.convert_to_tensor(self.batch_critic_outputs)

        actor_loss, actor_info  = self.compute_actor_loss(tf)
        critic_loss, critic_info = self.compute_critic_loss(tf)
        
        ret_info = {
            'actor_loss'    : actor_loss.numpy(),
            'critic_loss'   : critic_loss.numpy()
        }
        return actor_loss, critic_loss, ret_info

    def send_gradients(self, tape, actor_loss, critic_loss):
        gradients = [tape.gradient(actor_loss, self.actor.trainable_weights)]
        if (self.use_action_value_critic):
            gradients.append(tape.gradient(critic_loss, self.critic.trainable_weights))
        self.gradient_updates_queue.put(gradients)
        return gradients

    def send_results(self, tf, info):
        if (self.in_scheduling_mode):
            actor_loss     = info['actor_loss']
            critic_loss    = info['critic_loss']
            rew_mean       = info['reward']
            mcs_mean, prb_mean = self.environment.calculate_mean(None, self.batch_info)
        else:
            rew_mean = '-1'
            mcs_mean = '-1'
            prb_mean = '-1'
            actor_loss = '-1'
            critic_loss = '-1'
        additional_columns = self.environment.get_csv_result_policy_output(self.batch_info)

        with self.write_to_results_queue_lock:
            self.episode_number.value += 1
            self.results_queue.put([ [self.worker_num, rew_mean, mcs_mean, prb_mean, actor_loss, critic_loss], additional_columns ])


    def run_in_collecting_stats_mode(self):
        self.environment.setup(self.worker_num, self.total_workers)
        with self.successfully_started_worker.get_lock():
            self.successfully_started_worker.value += 1
        
        for self.ep_ix in range(self.episodes_to_run):
            self.batch_info = []
            if (self.ep_ix % 1 == 0):
                print(str(self) + ' -> Episode {}/{}'.format(self.ep_ix + 1, self.episodes_to_run))
            for _ in range(self.batch_size):
                next_state, reward, done, info = self.environment.step(None)
                self.batch_info.append(info)
            self.send_results(None, self.batch_info)


    def learn(self, tf):
        state_batch, action_batch, reward_batch = self.buffer.sample(tf)
        with tf.GradientTape() as tape1:
            critic_value = self.critic([state_batch, action_batch], training = True)
            critic_loss = tf.math.reduce_mean(tf.math.square(reward_batch - critic_value))
        critic_grads = tape1.gradient(critic_loss, self.critic.trainable_weights)

        with tf.GradientTape() as tape2:
            new_action_batch = self.actor(state_batch)
            actor_loss = -1 * self.critic([state_batch, new_action_batch])
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_grads = tape2.gradient(actor_loss, self.actor.trainable_weights)

        gradients = [actor_grads, critic_grads]
        self.gradient_updates_queue.put(gradients)
        info = {
            'actor_loss' : actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'reward'     : tf.math.reduce_mean(reward_batch).numpy()
        }
        return info


    def run(self) -> None:
        if (not self.in_scheduling_mode):
            self.run_in_collecting_stats_mode()
            return
        tf, _, self.tfp = import_tensorflow('3', True)
        self.tf = tf
        self.set_process_seeds(tf, self.worker_num)        
        self.initiate_worker_variables(tf)        
        try:
            self.initiate_models(tf)
            self.environment.setup(self.worker_num, self.total_workers)
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            for self.ep_ix in range(self.episodes_to_run):
                if (self.ep_ix % 1 == 0):
                    print(str(self) + ' -> Episode {}/{}'.format(self.ep_ix + 1, self.episodes_to_run))
                if (self.ep_ix % self.local_update_period == 0):
                    self.update_weights()

                self.batch_info = []

                batch_idx = 0
                while batch_idx < self.batch_size:
                    state = self.environment.reset()
                    done = False
                    while not done:
                        action = self.pick_action_from_embedding_table(state, tf)
                        next_state, reward, done, info = self.environment.step([action])

                        self.buffer.record((state, action, reward))
                        self.batch_info.append(info)
                        state = next_state
                        batch_idx += 1

                info = self.learn(tf)                
                self.send_results(tf, info)                
                
        finally:
            print(str(self) + ' -> Exiting...')

    def pick_action_and_get_critic_values(self, state: np.array, tf):
        tensor_state = tf.convert_to_tensor([state], dtype = tf.float32)
        
        model_output = self.actor(tensor_state)

        if (not self.use_action_value_critic):
            actor_output_probs = model_output[0]
            actor_output_log_probs = tf.math.log(actor_output_probs + 1e-10)
            actor_sample = tf.random.categorical(actor_output_log_probs, num_samples = 1) 
            action = [actor_sample[0][0].numpy()]
            critic_output = model_output[1]
            info = []
        elif (self.use_action_value_critic):
            actor_output_probs = model_output
            actor_output_log_probs = tf.math.log(actor_output_probs + 1e-10)
            actor_sample = tf.random.categorical(actor_output_log_probs, num_samples = 1) 
            action = [actor_sample[0][0].numpy()]
            
            tensor_state = tf.convert_to_tensor([state])
            critic_output_distr = self.critic(tensor_state) # distribution
            critic_output = critic_output_distr[0]
            info = []

        return action, actor_output_log_probs, critic_output, info

    def pick_action_from_embedding_table(self, state: np.array, tf):
        actor_input = tf.convert_to_tensor([state], dtype = tf.float32)
        model_output = self.actor(actor_input, training = True)

        noise = self.ou_noise()
        tbs = 24496 * model_output + noise
        
        k_closest_actions = self.environment.get_k_closest_actions(self.k, tbs)
        k_closest_actions_tensor = tf.expand_dims(tf.convert_to_tensor(k_closest_actions, dtype = tf.float32), axis = 1)
        actor_input_k_repeated = tf.repeat(actor_input, self.k, axis = 0)
        critic_outputs = self.critic([actor_input_k_repeated, k_closest_actions_tensor], training = True)

        critic_arg_max = tf.math.argmax(input = critic_outputs)[0].numpy()

        return k_closest_actions[critic_arg_max]



