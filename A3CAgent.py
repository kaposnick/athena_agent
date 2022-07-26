from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import random
import gym
import numpy as np
import BaseAgent
import multiprocessing as mp
from BaseAgent import BaseAgent
from WorkerAgent import Actor_Critic_Worker
from MasterAgent import Master_Agent
from common_utils import MODE_INFERENCE, MODE_SCHEDULING_AC, MODE_SCHEDULING_NO, MODE_SCHEDULING_RANDOM, MODE_TRAINING, import_tensorflow

COL_DT_AC_LOSS_MEAN = 'ac_loss_mean'
COL_DT_CR_LOSS_MEAN = 'cr_loss_mean'
COL_DT_ENTROPY_MEAN = 'entropy'
COL_DT_MCS_MEAN = 'mcs_mean'
COL_DT_PRB_MEAN = 'prb_mean'
COL_DT_REP = 'rep'
COL_DT_REWARD_MEAN = 'reward_mean'
COL_DT_SIM = 'sim'

COLUMNS = [
    COL_DT_REP, COL_DT_SIM,
    COL_DT_REWARD_MEAN, 'actor_nll', COL_DT_ENTROPY_MEAN, COL_DT_MCS_MEAN, COL_DT_PRB_MEAN, 'tbs_mean', 'tbs_stdv', COL_DT_AC_LOSS_MEAN, COL_DT_CR_LOSS_MEAN
]

class A3CAgent(BaseAgent):
    def __init__(self, config, num_processes, scheduling_mode = MODE_SCHEDULING_AC, training_mode = MODE_TRAINING) -> None:
        super(A3CAgent, self).__init__(config)  
        self.num_processes = num_processes
        self.worker_processes = []
        self.optimizer_worker = None
        self.actor_memory_name = 'model_actor'
        self.critic_memory_name = 'model_critic'
        self.scheduling_mode = scheduling_mode
        self.training_mode = training_mode

    def kill_all(self):
        self.exit_gracefully()

    def create_shared_memory(self, memory_name, memory_size):
        try:
            shm = shared_memory.SharedMemory(name = memory_name, create = True , size = memory_size)
        except:
            shm = shared_memory.SharedMemory(name = memory_name, create = False, size = memory_size)
        return shm

    def run_in_collect_stats_mode(self, processes_started_successfully = None, inputs = None):
        successfully_started_worker = mp.Value('i', 0)
        for worker_num in range(self.num_processes):
            import copy
            worker_environment = copy.deepcopy(self.environment)
            if (inputs is not None):
                worker_environment.presetup(inputs[worker_num])
            worker = Actor_Critic_Worker(
                worker_environment, self.config, 
                worker_num, self.num_processes,
                None, None,
                successfully_started_worker,
                None, self.write_to_results_queue_lock,
                self.results_queue, None,
                self.episode_number, 
                scheduling_mode=False)
            worker.start()
            self.worker_processes.append(worker)
        while (successfully_started_worker.value < self.num_processes):
            pass
        if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
        if (self.config.save_results):
            self.save_results(self.config.results_file_path, self.results_queue)
        for worker in self.worker_processes:
            worker.join()

    def execute_in_random_scheduling_mode(self, processes_started_successfully, inputs):
        sample_buffer_queue = mp.Queue()
        master_agent_initialized = mp.Value('i', 0)
        master_agent_stop = mp.Value('i', 0)
        worker_agent_initialized = mp.Value('i', 0)
        worker_agent_stop = mp.Value('i', 0)

        master_agent = Master_Agent(
            self.environment,
            self.hyperparameters, self.config, self.state_size, self.action_size,
            None, None, None,
            master_agent_initialized=master_agent_initialized,
            sample_buffer_queue=sample_buffer_queue,
            batch_info_queue=None,
            results_queue=None,
            master_agent_stop=master_agent_stop,
            optimizer_lock=None,
            in_training_mode=None,
            scheduling_mode=self.scheduling_mode
        )
        master_agent.start()
        while (master_agent_initialized.value == 0):
                pass
        print('Optimizer thread started successfully')
        for worker_num in range(self.num_processes):
            import copy
            worker_environment = copy.deepcopy(self.environment)
            if (inputs is not None):
                worker_environment.presetup(inputs[worker_num])
            worker = Actor_Critic_Worker(
                worker_environment, self.config, 
                worker_num, self.num_processes,
                state_size = self.state_size, action_size = self.action_size,
                successfully_started_worker=worker_agent_initialized,
                sample_buffer_queue = sample_buffer_queue, 
                batch_info_queue = None, optimizer_lock=None,
                scheduling_mode=self.scheduling_mode, 
                in_training_mode =None, 
                worker_agent_stop_value=worker_agent_stop)
            worker.start()
            self.worker_processes.append(worker)
        while (worker_agent_initialized.value < self.num_processes):
            pass
        if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
        import time
        seconds = 7200
        print('A3C Agent going to sleep for {} seconds...'.format(seconds))
        time.sleep(seconds)
        print('A3C Agent woke up.. Finalizing master agent thread')
        master_agent_stop.value = 1
        master_agent.join()
        


    def run_n_episodes(self, processes_started_successfully = None, inputs = None):
        self.episode_number                    = mp.Value('i', 0)        
        self.results_queue                     = mp.Queue()  
        
        if (self.scheduling_mode == MODE_SCHEDULING_NO):
            self.run_in_collect_stats_mode(processes_started_successfully, inputs)
            return
        elif (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            self.execute_in_random_scheduling_mode(processes_started_successfully, inputs)
            return

        
        sample_buffer_queue               = mp.Queue()
        batch_info_queue                  = mp.Queue()
        optimizer_lock                    = mp.Lock()
        self.in_training_mode             = mp.Value('i', self.training_mode)
        actor_memory_size_in_bytes        = mp.Value('i', 0)
        critic_memory_size_in_bytes       = mp.Value('i', 0)
        memory_created                    = mp.Value('i', 0)
        master_agent_initialized          = mp.Value('i', 0)
        master_agent_stop                 = mp.Value('i', 0)
        worker_agent_stop                 = mp.Value('i', 0)
        successfully_started_worker       = mp.Value('i', 0)

        self.use_action_value_critic = not self.config.hyperparameters['Actor_Critic_Common']['use_state_value_critic']
        self.optimizer_worker = Master_Agent(
                environment = self.environment,
                hyperparameters = self.hyperparameters,
                config = self.config, state_size = self.state_size, action_size = self.action_size,
                actor_memory_bytes = actor_memory_size_in_bytes, critic_memory_bytes = critic_memory_size_in_bytes,
                memory_created = memory_created, master_agent_initialized = master_agent_initialized,
                sample_buffer_queue = sample_buffer_queue, batch_info_queue = batch_info_queue,
                results_queue = self.results_queue,
                master_agent_stop = master_agent_stop,
                optimizer_lock = optimizer_lock,
                in_training_mode = self.in_training_mode,
                scheduling_mode = self.scheduling_mode,
                actor_memory_name = self.actor_memory_name, 
                critic_memory_name = self.critic_memory_name
            )
        
        
        self.optimizer_worker.start()
        while (actor_memory_size_in_bytes.value == 0):
            pass

        while (self.use_action_value_critic and critic_memory_size_in_bytes.value == 0):
            pass

        try:
            self.shm_actor  = self.create_shared_memory(self.actor_memory_name ,  actor_memory_size_in_bytes.value)

            if (self.use_action_value_critic):
                self.shm_critic = self.create_shared_memory(self.critic_memory_name, critic_memory_size_in_bytes.value)
            memory_created.value = 1
                        
            while (master_agent_initialized.value == 0):
                pass
            
            save_file = self.config.results_file_path
            print('Optimizer thread started successfully')
            for worker_num in range(self.num_processes):
                import copy
                worker_environment = copy.deepcopy(self.environment)
                if (inputs is not None):
                    worker_environment.presetup(inputs[worker_num])
                worker = Actor_Critic_Worker(
                    environment = worker_environment, 
                    config = self.config, 
                    worker_num = worker_num, total_workers = self.num_processes,
                    state_size = self.state_size, action_size = self.action_size,
                    successfully_started_worker=successfully_started_worker,
                    sample_buffer_queue=sample_buffer_queue, batch_info_queue=batch_info_queue,
                    optimizer_lock = optimizer_lock, scheduling_mode=self.scheduling_mode, 
                    in_training_mode = self.in_training_mode, 
                    worker_agent_stop_value=worker_agent_stop)
                worker.start()
                self.worker_processes.append(worker)
            while (successfully_started_worker.value < self.num_processes):
                pass

            if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
            if (self.config.save_results):
                self.save_results(save_file, self.results_queue)
            
            # worker_agent_stop.value = 1
            # for worker in self.worker_processes:
            #     worker.join()

            master_agent_stop.value = 1
            self.optimizer_worker.join()
        finally:
            self.exit_gracefully()


    def save_results(self, save_file, results_queue):
        with open(save_file, 'w') as f:
            additional_env_columns = self.environment.get_csv_result_policy_output_columns()
            f.write('|'.join(COLUMNS + additional_env_columns) + '\n')            
            episode = 0
            has_entered_inference_mode = False
            while True:
                carry_on = True
                if (episode == self.config.num_episodes_to_run and not has_entered_inference_mode):
                    self.in_training_mode.value = MODE_INFERENCE
                    has_entered_inference_mode = True
                    print('Going to inference mode')
                if (episode == self.config.num_episodes_to_run + self.config.num_episodes_inference):
                    print('Finishing printing results...')
                    carry_on = False
                if carry_on:
                    if not results_queue.empty():
                        episode += 1
                        q_result = results_queue.get()
                        result = [str(x) for x in [episode, *q_result[0]]]
                        for additional_column in q_result[1]:
                            period = additional_column['period']
                            value  = additional_column['value']
                            if (episode % period == 0):
                                result.append(str(value))
                            else:
                                result.append('')
                        f.write('|'.join(result) + '\n')
                        f.flush()
                else: break

    def exit_gracefully(self):
        print('Killing worker processes...')
        for worker_process in self.worker_processes:
            if (worker_process.is_alive()):
                worker_process.kill()
                worker_process.join()
        
        print('Killing master process')
        if (self.optimizer_worker.is_alive()):
            self.optimizer_worker.kill()
            self.optimizer_worker.join()       