import multiprocessing as mp
import numpy as np
import random
from common_utils import MODE_SCHEDULING_ATHENA, MODE_SCHEDULING_RANDOM, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer, MCS_SPACE, PRB_SPACE
from agent_ddpg import DDPGAgent
from srsran_env import SrsRanEnv

import copy

class HarqAgent(mp.Process):
    def __init__(self, 
                environment: SrsRanEnv, 
                worker_num: np.int32, total_workers: np.int32, 
                context_size: np.int32, action_size: np.int32, 
                successfully_started_worker: mp.Value,
                results_queue: mp.Queue,
                scheduling_mode: str,
                verbose: int = 0,
                actor_memory_name: str = 'model_actor',
                critic_memory_name: str = 'model_critic') -> None:
        super(HarqAgent, self).__init__()
        # environment variables
        self.environment = environment
        self.worker_num = worker_num
        self.total_workers = total_workers
        self.context_size = context_size
        self.action_size = action_size

        # multiprocessing variables
        self.successfully_started_worker = successfully_started_worker
        self.results_queue = results_queue
        self.scheduling_mode = scheduling_mode
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name    

        self.verbose = verbose

    def set_process_seeds(self, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(worker_num)
        random.seed(worker_num)
        np.random.seed(worker_num)
        if (hasattr(self, 'tf')):
            self.tf.random.set_seed(worker_num)

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
            self.ddpg_agent = DDPGAgent(self.tf, self.context_size, self.action_size)
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
            self.print_verbose('Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def update_weights(self):
        self.ddpg_agent.actor.set_weights(copy.deepcopy(self.weights_actor))            
        self.ddpg_agent.critic.set_weights(copy.deepcopy(self.weights_critic))

    def run(self):
        self.tf, _, self.tfp = import_tensorflow('3', False)
        self.set_process_seeds(self.worker_num)
        try:
            associate_with_master = self.scheduling_mode == MODE_SCHEDULING_ATHENA
            self.initiate_models(associate_with_master=associate_with_master)
            self.environment.setup(self.worker_num, self.total_workers)

            if (associate_with_master):
                self.update_weights()
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            print('HARQ Agent ' + str(self.worker_num) + ' initialized')
            while (True):
                environment_context = self.environment.reset()
                context = environment_context.copy()
                action, mcs, prb = self.ddpg_agent(context)

                if (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
                    mcs = MCS_SPACE[np.random.randint(0, len(MCS_SPACE))]
                    prb = PRB_SPACE[np.random.randint(0, len(PRB_SPACE))]
                
                _, reward, _, info = self.environment.step([mcs, prb])
                if (reward is None):
                    # This happens in cases where the srsRAN doesn't apply the decided action
                    # As a result, no reward is being returned from the environment, and we 
                    # don't want to record this sample on the record file, neither the MasterAgent
                    # to learn from this experience.
                    self.print_verbose('Action not applied.. skipping')
                    continue

                if (self.environment.is_context_valid()):                    
                    info['mu'] = mcs
                    info['sigma'] = prb
                    self.results_queue.put(info)
                else:
                    print('non valid')

        except Exception as e:
            print(str(self) + str(e))     
        finally:
            print(str(self) + ' -> Exiting...')

    def print_verbose(self, string_to_print, end = None):
        if (self.verbose == 1 and self.worker_num == 0):
            print(str(self) + ' -> ' + string_to_print, end=end)

    def __str__(self) -> str:
        return 'Worker ' + str(self.worker_num)



