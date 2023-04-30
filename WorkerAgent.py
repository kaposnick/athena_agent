import multiprocessing as mp
import numpy as np
import random
from OUActionNoise import OUActionNoise
from env.BaseEnv import BaseEnv
from common_utils import MODE_SCHEDULING_AC, MODE_SCHEDULING_NO, MODE_SCHEDULING_RANDOM, MODE_TRAINING, get_basic_actor_network, get_basic_critic_network, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer, normalize_state, normalize_mcs_prb, denormalize_mcs_prb, denormalize_mcs
from DDPGAgent import DDPGAgent
from Config import Config
import time
import copy
from scipy.spatial import distance

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
        variables = np.sum([np.prod(v.shape) for v in model.variables])
        size = variables * model_dtype.itemsize
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initiate_models(self, associate_with_master=True):
        try:
            stage = 'Creating the neural networks'
            self.ddpg_agent = DDPGAgent(self.tf, self.state_size, 1)
            self.ddpg_agent.set_action_array(self.environment.action_array)
            self.ddpg_agent.load_actor() 
            self.ddpg_agent.load_critic()

            stage = 'Actor memory reference creation'
            if (associate_with_master):
                self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.ddpg_agent.actor, self.actor_memory_name)
                self.weights_actor = map_weights_to_shared_memory_buffer(self.ddpg_agent.actor.get_weights(), self.np_array_actor)

            stage = 'Action-value critic memory reference creation'
            if (associate_with_master):
                self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.ddpg_agent.critic, self.critic_memory_name)
                self.weights_critic = map_weights_to_shared_memory_buffer(self.ddpg_agent.critic.get_weights(), self.np_array_critic)
        except Exception as e:
            self.print('Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def update_weights(self):
        with self.optimizer_lock:
            self.ddpg_agent.actor.set_weights(copy.deepcopy(self.weights_actor))            
            self.ddpg_agent.critic.set_weights(copy.deepcopy(self.weights_critic))

    def execute_in_collecting_stats_mode(self):
        self.environment.setup(self.worker_num, self.total_workers)
        with self.successfully_started_worker.get_lock():
            self.successfully_started_worker.value += 1
        
        while (True):
            state = self.environment.reset()
            _, reward, _, info = self.environment.step((24, 45))
            if (reward is None):
                continue
            if (self.environment.is_state_valid()):
                info['mu'] = -1
                info['sigma'] = -1
            self.batch_info_queue.put(info)

    def execute_in_schedule_random_mode(self):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds(self.worker_num)
        try:
            self.initiate_models(associate_with_master=False)
            self.environment.setup(self.worker_num, self.total_workers)        
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            while (True):
                state = self.environment.reset()
                action,mcs, prb = self.ddpg_agent(state)
                action = 'random'
                _, reward, _, info = self.environment.step([action])
                if (reward is None):
                    continue
                if (info['modified']):
                    # This may happen in cases where the srsRAN chooses to allocate fewer PRBs
                    # than the decides ones. IN this case we don't want to record this sample
                    # on the record file, and we don't want the MasterAgent to learn from this
                    # experience.
                    self.print('Modified...')
                    continue
                if (self.environment.is_state_valid()):
                    action = np.array([info['mcs'], info['prb']])
                    reward = np.array([info['crc'], info['decoding_time'], info['snr_decode'], info['noise_decode'], info['snr_custom'], info['gain']])
                    self.sample_buffer_queue.put([(state, action, reward)])
        finally:
            print(str(self) + ' -> Exiting...')


    def execute_in_schedule_ac_mode(self, associate_with_master=True):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds(self.worker_num)
        try:
            self.initiate_models(associate_with_master=associate_with_master)
            self.environment.setup(self.worker_num, self.total_workers)

            if (associate_with_master):
                self.update_weights()
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            self.ep_ix = 0
            while (True):
                self.ep_ix += 1                
                environment_state = self.environment.reset()
                state = environment_state.copy()
                state[1] = state[1] - 1.5
                action, mcs, prb = self.ddpg_agent(state)
                _, reward, _, info = self.environment.step([mcs, prb])
                if (reward is None):
                    # This happens in cases where the srsRAN doesn't apply the decided action
                    # As a result, no reward is being returned from the environment, and we 
                    # don't want to record this sample on the record file, neither the MasterAgent
                    # to learn from this experience.
                    self.print('Action not applied.. skipping')
                    continue
                if (info['modified'] and False):
                    # This may happen in cases where the srsRAN chooses to allocate fewer PRBs
                    # than the decides ones. IN this case we don't want to record this sample
                    # on the record file, and we don't want the MasterAgent to learn from this
                    # experience.
                    self.print('Modified...')
                    continue

                if (self.environment.is_state_valid()):
                    if (self.isin_training_mode()):
                        self.sample_buffer_queue.put([(state, action, reward)])                
                    info['mu'] = mcs
                    info['sigma'] = prb
                    self.batch_info_queue.put(info)
                
        finally:
            print(str(self) + ' -> Exiting...')

    def isin_training_mode(self):
        return self.scheduling_mode == MODE_SCHEDULING_AC and self.in_training_mode.value == MODE_TRAINING

    def run(self) -> None:
        if (self.scheduling_mode == MODE_SCHEDULING_NO):
            # self.execute_in_collecting_stats_mode()
            self.execute_in_schedule_ac_mode(associate_with_master=False)
        elif (self.scheduling_mode == MODE_SCHEDULING_AC):
            self.execute_in_schedule_ac_mode()
        elif (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            self.execute_in_schedule_random_mode()

    def print(self, string_to_print, end = None):
        if ((self.worker_num == 4 and False) or True):
            print(str(self) + ' -> ' + string_to_print, end=end)



