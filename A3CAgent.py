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
from common_utils import import_tensorflow

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
    COL_DT_REWARD_MEAN, COL_DT_MCS_MEAN, COL_DT_PRB_MEAN, COL_DT_AC_LOSS_MEAN, COL_DT_CR_LOSS_MEAN
]

class A3CAgent(BaseAgent):
    def __init__(self, config, num_processes, in_scheduling_mode = True) -> None:
        super(A3CAgent, self).__init__(config)  
        self.num_processes = num_processes
        self.worker_processes = []
        self.optimizer_worker = None
        self.actor_memory_name = 'model_actor'
        self.critic_memory_name = 'model_critic'
        self.in_scheduling_mode = in_scheduling_mode

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
                self.episodes_per_process, None, None,
                successfully_started_worker,
                None, self.write_to_results_queue_lock,
                self.results_queue, None,
                self.episode_number, 
                in_scheduling_mode=False)
            worker.start()
            self.worker_processes.append(worker)
        while (successfully_started_worker.value < self.num_processes):
            pass
        if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
        if (self.config.save_results):
            self.save_results(self.config.results_file_path, self.episode_number, self.results_queue)
        for worker in self.worker_processes:
            worker.join()

    def run_n_episodes(self, processes_started_successfully = None, inputs = None):
        if (self.config.num_episodes_to_run > 0):
            self.episodes_per_process = int(self.config.num_episodes_to_run / self.num_processes)
        else:
            self.episodes_per_process = -1 # run indefinitely
        self.episode_number                    = mp.Value('i', 0)        
        self.results_queue                     = mp.Queue()
        self.write_to_results_queue_lock  = mp.Lock()        
        
        if (not self.in_scheduling_mode):
            self.run_in_collect_stats_mode(processes_started_successfully, inputs)
            return
        
        gradient_updates_queue            = mp.Queue()
        optimizer_lock                    = mp.Lock()
        actor_memory_size_in_bytes        = mp.Value('i', 0)
        critic_memory_size_in_bytes       = mp.Value('i', 0)
        memory_created                    = mp.Value('i', 0)
        master_agent_initialized          = mp.Value('i', 0)
        master_agent_stop                 = mp.Value('i', 0)
        successfully_started_worker       = mp.Value('i', 0)

        self.use_action_value_critic = not self.config.hyperparameters['Actor_Critic_Common']['use_state_value_critic']
        self.optimizer_worker = Master_Agent(
                self.hyperparameters,
                self.config, self.state_size, self.action_size,
                actor_memory_size_in_bytes, critic_memory_size_in_bytes,
                memory_created, master_agent_initialized,
                gradient_updates_queue, master_agent_stop,
                optimizer_lock,
                self.actor_memory_name, self.critic_memory_name
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
                    worker_environment, self.config, 
                    worker_num, self.num_processes,
                    self.episodes_per_process, self.state_size, self.action_size,
                    successfully_started_worker,
                    optimizer_lock, self.write_to_results_queue_lock,
                    self.results_queue, gradient_updates_queue,
                    self.episode_number, in_scheduling_mode=True)
                worker.start()
                self.worker_processes.append(worker)
            while (successfully_started_worker.value < self.num_processes):
                pass

            if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
            if (self.config.save_results):
                self.save_results(save_file, self.episode_number, self.results_queue)
            for worker in self.worker_processes:
                worker.join()
            master_agent_stop.value = 1
            self.optimizer_worker.join()
        finally:
            self.exit_gracefully()


    def save_results(self, save_file, episode_number, results_queue):
        with open(save_file, 'w') as f:
            additional_env_columns = self.environment.get_csv_result_policy_output_columns()
            f.write('|'.join(COLUMNS + additional_env_columns) + '\n')            
            while True:
                with self.write_to_results_queue_lock:
                    carry_on = episode_number.value < self.config.num_episodes_to_run
                    episode = episode_number.value
                if carry_on:
                    if not results_queue.empty():
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
        for worker_process in self.worker_processes:
            if (worker_process.is_alive()):
                worker_process.kill()
                worker_process.join()
        
        if (self.optimizer_worker.is_alive()):
            self.optimizer_worker.kill()
            self.optimizer_worker.join()       