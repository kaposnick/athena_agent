import multiprocessing as mp
import numpy as np
import random
from OUActionNoise import OUActionNoise
from Buffer import EpisodeBuffer
from env.BaseEnv import BaseEnv
from common_utils import MODE_SCHEDULING_AC, MODE_SCHEDULING_NO, MODE_SCHEDULING_RANDOM, MODE_TRAINING, get_basic_actor_network, get_basic_critic_network, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer, normalize_state
from Config import Config
import time
import copy

class Actor_Critic_Worker(mp.Process):
    def __init__(self, 
                environment: BaseEnv, config: Config,
                worker_num: np.int32, total_workers: np.int32, 
                state_size, action_size, 
                successfully_started_worker: mp.Value,
                sample_buffer_queue: mp.Queue, batch_info_queue: mp.Queue, 
                optimizer_lock: mp.Lock, 
                scheduling_mode: str,
                in_training_mode: mp.Value,
                worker_agent_stop_value = mp.Value,
                actor_memory_name = 'model_actor',
                critic_memory_name = 'model_critic') -> None:
        super(Actor_Critic_Worker, self).__init__()
        # environment variables
        self.environment = environment
        self.config = config
        self.worker_num = worker_num
        self.total_workers = total_workers
        self.state_size = state_size
        self.action_size = action_size

        # multiprocessing variables
        self.successfully_started_worker = successfully_started_worker
        self.optimizer_lock = optimizer_lock
        self.sample_buffer_queue = sample_buffer_queue
        self.batch_info_queue = batch_info_queue
        self.scheduling_mode = scheduling_mode
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name

        self.in_training_mode = in_training_mode
        self.worker_agent_stop_value = worker_agent_stop_value

        self.init_configuration()        

    def __str__(self) -> str:
        return 'Worker ' + str(self.worker_num)
        
    def init_configuration(self):
        self.local_update_period = self.config.hyperparameters['Actor_Critic_Common']['local_update_period']    

    def set_process_seeds(self, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed + worker_num)
        random.seed(self.config.seed + worker_num)
        np.random.seed(self.config.seed + worker_num)
        if (hasattr(self, 'tf')):
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
        self.coef = 5.159817058590249
        self.intercept = 7.6586417701293446


    def update_weights(self):
        with self.optimizer_lock:
            self.actor.set_weights(copy.deepcopy(self.weights_actor))            
            self.critic.set_weights(copy.deepcopy(self.weights_critic))


    def send_results(self, info):
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

        with self.write_to_results_queue_lock:
            self.results_queue.put([ [self.worker_num, rew_mean, actor_nll, entropy, mcs_mean, prb_mean, tbs_mean, tbs_stdv, actor_loss, critic_loss], additional_columns ])

    def convert_to_real_action_applied(self, info):
        mcs = int(info['mcs'])
        prb = int(info['prb'])
        tbs = self.environment.to_tbs(mcs, prb)
        return tbs
        return (np.log(tbs + 1) - self.intercept) / self.coef

    def run_in_collecting_stats_mode(self):
        self.environment.setup(self.worker_num, self.total_workers)
        with self.successfully_started_worker.get_lock():
            self.successfully_started_worker.value += 1
        
        
        self.ep_ix = 0
        while (True):
            self.ep_ix += 1

            self.batch_info = []
            next_state, reward, done, info = self.environment.step(None)
            if (reward is None):
                state = next_state
                continue
            self.batch_info.append(info)
            state = next_state
            self.batch_info_queue.put(self.batch_info)

    def execute_in_schedule_random_mode(self):
        self.set_process_seeds(self.worker_num)
        try:
            self.environment.setup(self.worker_num, self.total_workers)        
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            ep_ix = 0
            while (True):
                ep_ix += 1
                if (ep_ix % 1 == 0):
                    self.print('Episode {}'.format(ep_ix + 1))
                sample_buffer = []
                state = self.environment.reset()
                next_state, reward, done, info = self.environment.step('random')
                if (reward is None):
                    continue
                action = np.array([info['mcs'], info['prb']])
                reward = np.array([info['crc'], info['decoding_time']])
                sample = (state, action, reward)
                sample_buffer.append(sample)
                
                self.sample_buffer_queue.put(sample_buffer)
        finally:
            print(str(self) + ' -> Exiting...')


    def execute_in_schedule_ac_mode(self):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds(self.worker_num)        
        self.initiate_worker_variables()        
        try:
            self.initiate_models()
            self.environment.setup(self.worker_num, self.total_workers)

            self.update_weights()
            self.print('Getting initial weights weights time')
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            self.ep_ix = 0
            while (True):
                self.ep_ix += 1
                if (self.ep_ix % 1 == 0):
                    self.print('Episode {}'.format(self.ep_ix + 1))
                if (self.in_training_mode.value == MODE_TRAINING and (self.ep_ix % self.local_update_period == 0)):
                    update_weights_time = time.time()
                    self.update_weights()
                    self.print('Updating weights time: {}'.format(time.time() - update_weights_time))

                self.batch_info = []
                self.sample_buffer = []
                
                state = self.environment.reset()
                done = False
                while not done:
                    pick_action_time = time.time()
                    action_idx, action, mu, sigma = self.pick_action_from_embedding_table(state)

                    step_time = time.time()
                    next_state, reward, done, info = self.environment.step([action_idx])
                    if (reward is None):
                        self.print('Action not applied.. skipping')
                        continue
                    if ('modified' in info and info['modified'] is True):
                        self.print('Modified...')
                        continue


                    real_action_applied = self.convert_to_real_action_applied(info)
                    info['mu'] = mu
                    info['sigma'] = sigma

                    is_state_valid = self.environment.is_valid(state)
                    if (is_state_valid and self.in_training_mode.value == MODE_TRAINING):
                        self.sample_buffer.append((normalize_state(state), action, reward))
                    
                    if (is_state_valid):
                        self.batch_info.append(info)
                    state = next_state

                if (self.in_training_mode.value == MODE_TRAINING and len(self.sample_buffer) > 0):
                    self.sample_buffer_queue.put(self.sample_buffer)
                
                if (len(self.batch_info) > 0):
                    self.batch_info_queue.put(self.batch_info)   
                
        finally:
            print(str(self) + ' -> Exiting...')


    def run(self) -> None:
        if (self.scheduling_mode == MODE_SCHEDULING_NO):
            self.run_in_collecting_stats_mode()
        elif (self.scheduling_mode == MODE_SCHEDULING_AC):
            self.execute_in_schedule_ac_mode()
        elif (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            self.execute_in_schedule_random_mode()

    def pick_action_from_embedding_table(self, state: np.array):
        in_training_mode = self.in_training_mode.value == MODE_TRAINING
        state = normalize_state(state)
        self.print('{}'.format(state))
        actor_input = self.tf.convert_to_tensor([state], dtype = self.tf.float32)
        mu, sigma = self.actor(actor_input)[0]
        sigma = 1e-5 + self.tf.math.softplus(sigma)

        if (in_training_mode):
            heta_param = np.random.normal(loc = 0, scale = 1)
            action_hat = mu + heta_param * sigma
        else:
            heta_param = np.random.normal(loc = 0, scale = 1)
            action_hat = mu
        # action_hat = mean

        tbs_hat = action_hat
        
        action_idx, tbs = self.environment.get_closest_actions(tbs_hat)
        action = tbs
        return action_idx, action, mu.numpy(), sigma.numpy()

    def print(self, string_to_print, end = None):
        if (self.worker_num == 4 and False):
            print(str(self) + ' -> ' + string_to_print, end=end)



