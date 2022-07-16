import multiprocessing as mp
import numpy as np
import random
from OUActionNoise import OUActionNoise
from Buffer import Buffer
from env.BaseEnv import BaseEnv
from common_utils import get_basic_actor_network, get_basic_critic_network, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer
from Config import Config
import time

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
        self.include_entropy_term = self.config.hyperparameters['Actor_Critic_Common']['include_entropy_term']
        self.entropy_contribution = self.config.hyperparameters['Actor_Critic_Common']['entropy_contribution']

    

    def set_process_seeds(self, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed + worker_num)
        random.seed(self.config.seed + worker_num)
        np.random.seed(self.config.seed + worker_num)
        self.tf.random.set_seed(self.config.seed + worker_num)

    def get_shared_memory_reference(self, model, memory_name):
        ## return the reference to the memory and the np.array pointing to the shared memory
        model_dtype = np.dtype(model.dtype)
        variables = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        size = variables * model_dtype.itemsize
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initiate_models(self):
        try:
            stage = 'Creating the neural networks'
            
            models = [
                get_basic_actor_network(self.tf, self.tfp, self.state_size), 
                get_basic_critic_network(self.tf, self.state_size, 1)
            ]

            stage = 'Actor memory reference creation'
            self.actor = models[0]
            self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.actor, self.actor_memory_name)
            self.weights_actor = map_weights_to_shared_memory_buffer(self.actor.get_weights(), self.np_array_actor)

            stage = 'Action-value critic memory reference creation'
            self.critic = models[1]
            self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.critic, self.critic_memory_name)
            self.weights_critic = map_weights_to_shared_memory_buffer(self.critic.get_weights(), self.np_array_critic)
        except Exception as e:
            self.print('Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def initiate_worker_variables(self):
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        self.k = 1
        self.buffer = Buffer(self.batch_size, self.batch_size, self.state_size, 1)
        self.coef = 5.159817058590249
        self.intercept = 7.6586417701293446

    def update_weights(self):
        import copy
        with self.optimizer_lock:
            self.actor.set_weights(copy.deepcopy(self.weights_actor))            
            self.critic.set_weights(copy.deepcopy(self.weights_critic))

    def send_gradients(self, tape, actor_loss, critic_loss):
        gradients = [tape.gradient(actor_loss, self.actor.trainable_weights)]        
        gradients.append(tape.gradient(critic_loss, self.critic.trainable_weights))
        self.gradient_updates_queue.put(gradients)
        return gradients

    def send_results(self, info):
        if (self.in_scheduling_mode):
            actor_loss     = info['actor_loss']
            critic_loss    = info['critic_loss']
            rew_mean       = info['reward']
            entropy        = info['entropy']
            actor_nll      = info['actor_nll']
            tbs_mean       = info['action_mean']
            tbs_stdv       = info['action_stdv']
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
            self.results_queue.put([ [self.worker_num, rew_mean, actor_nll, entropy, mcs_mean, prb_mean, tbs_mean, tbs_stdv, actor_loss, critic_loss], additional_columns ])


    def run_in_collecting_stats_mode(self):
        self.environment.setup(self.worker_num, self.total_workers)
        with self.successfully_started_worker.get_lock():
            self.successfully_started_worker.value += 1
        
        for self.ep_ix in range(self.episodes_to_run):
            self.batch_info = []
            if (self.ep_ix % 1 == 0):
                self.print('Episode {}/{}'.format(self.ep_ix + 1, self.episodes_to_run))
            for _ in range(self.batch_size):
                next_state, reward, done, info = self.environment.step(None)
                self.batch_info.append(info)
            self.send_results(None, self.batch_info)


    def learn(self):
        state_batch, action_batch, reward_batch = self.buffer.sample(self.tf)
        with self.tf.GradientTape() as tape1, self.tf.GradientTape() as tape2:
            distr_batch    =  self.actor(state_batch, training = True)
            critic_batch   = self.critic([state_batch, action_batch], training = True)
            advantage = reward_batch - critic_batch
            
            nll = distr_batch.log_prob(action_batch)
            # actor_loss = -1 * self.tf.math.reduce_mean(nll * advantage)
            actor_advantage_nll = nll * (reward_batch - self.tf.math.exp(nll) * critic_batch)
            actor_loss = actor_advantage_nll

            entropy = distr_batch.entropy()
            if (self.include_entropy_term):
                actor_loss += self.entropy_contribution * entropy
                        
            actor_loss  = -1 * self.tf.math.reduce_mean(actor_loss)
            critic_loss = self.tf.math.reduce_mean(self.tf.math.square(advantage))
        
        actor_grads  = tape1.gradient(actor_loss, self.actor.trainable_weights)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_weights)        

        gradients = [actor_grads, critic_grads]
        self.gradient_updates_queue.put(gradients)
        info = {
            'actor_loss' : actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'reward'     : self.tf.math.reduce_mean(reward_batch).numpy(),
            'entropy'    : self.tf.math.reduce_mean(entropy).numpy(),
            'actor_nll'  : self.tf.math.reduce_mean(actor_advantage_nll).numpy(),
            'action_mean': self.tf.math.reduce_mean(distr_batch.mean()).numpy(),
            'action_stdv': self.tf.math.reduce_mean(distr_batch.stddev()).numpy()
        }
        return info

    def run(self) -> None:
        if (not self.in_scheduling_mode):
            self.run_in_collecting_stats_mode()
            return
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds(self.worker_num)        
        self.initiate_worker_variables()        
        try:
            self.initiate_models()
            self.environment.setup(self.worker_num, self.total_workers)
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            for self.ep_ix in range(self.episodes_to_run):
                if (self.ep_ix % 1 == 0):
                    self.print('Episode {}/{}'.format(self.ep_ix + 1, self.episodes_to_run))
                if (self.ep_ix % self.local_update_period == 0):
                    update_weights_time = time.time()
                    self.update_weights()
                    self.print('Updating weights time: {}'.format(time.time() - update_weights_time))

                self.batch_info = []

                batch_idx = 0
                while batch_idx < self.batch_size:
                    self.print('Batch idx {}'.format(batch_idx))
                    
                    wait_state_time = time.time()
                    state = self.environment.reset()
                    self.print('Wait for state time: {}'.format(time.time() - wait_state_time))
                    done = False
                    while not done:
                        pick_action_time = time.time()
                        action_idx, action = self.pick_action_from_embedding_table(state)
                        self.print('Pick action time: {}'.format(time.time() - pick_action_time))

                        step_time = time.time()
                        next_state, reward, done, info = self.environment.step([action_idx])
                        self.print('Executing action time: {}'.format(time.time() - step_time))

                        self.buffer.record((state, action, reward))
                        self.batch_info.append(info)
                        state = next_state
                        batch_idx += 1

                learning_time = time.time()
                info = self.learn()    
                self.print('Learning time: {}'.format(time.time() - learning_time))

                self.print('Sending results...')            
                sending_results_time = time.time()
                self.send_results(info)           
                self.print('Sending results time: {}'.format(time.time() - sending_results_time))     
                
        finally:
            self.print('Exiting...')

    def pick_action_from_embedding_table(self, state: np.array):
        actor_input = self.tf.convert_to_tensor([state], dtype = self.tf.float32)
        distr = self.actor(actor_input)

        heta_param = np.random.normal(loc = 0, scale = 1)
        mean = distr.mean()
        stddev = distr.stddev()
        action_hat = mean + heta_param * stddev
        # action_hat = mean

        tbs_hat = np.exp(self.coef * action_hat + self.intercept) - 1
        
        k_closest_actions = self.environment.get_k_closest_actions(self.k, tbs_hat)

        action_idx = k_closest_actions[0]
        tbs        = k_closest_actions[1]

        action = (np.log(tbs + 1) - self.intercept) / self.coef
        self.print('tbs hat: {} - tbs: {}'.format(tbs_hat, tbs))
        self.print('action hat: {}/{}/{} - action: {}'.format(mean[0][0].numpy(), stddev[0][0].numpy(), action_hat, action))
        return action_idx, action

    def print(self, string_to_print, end = None):
        if (self.worker_num == 2):
            print(str(self) + ' -> ' + string_to_print, end=end)



