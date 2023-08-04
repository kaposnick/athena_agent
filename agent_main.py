import multiprocessing as mp
import numpy as np

from common_utils import get_shared_memory_ref, import_tensorflow, map_weights_to_shared_memory_buffer, publish_weights_to_shared_memory
from agent_ddpg import DDPGAgent

class MainAgent(mp.Process):
    def __init__(self,context_size, action_size,
                load_initial_weights,
                main_agent_initialized, stop_flag,
                actor_initial_weights_path=None,
                critic_initial_weights_path=None,
                actor_memory_name = 'model_actor',
                critic_memory_name = 'model_critic') -> None:
        super(MainAgent, self).__init__()
        self.context_size = context_size
        self.action_size = action_size
        self.load_initial_weights=load_initial_weights
        self.actor_initial_weights_path=actor_initial_weights_path
        self.critic_initial_weights_path=critic_initial_weights_path
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name
        self.main_agent_initialized = main_agent_initialized
        self.stop_flag = stop_flag

    def run(self):
        import signal
        signal.signal(signal.SIGINT , self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.tf, _, _ = import_tensorflow('3', False)
        try:
            self.initialize_models()
            self.load_weights()           
            self.publish_weights()
            self.main_agent_initialized.value = 1
            import time
            while(self.stop_flag.value == 0):
                time.sleep(3)
        finally:
            print(str(self) + ' -> Exiting...')

    def exit_gracefully(self, signum, frame):
        self.stop_flag.value = 1

    def compute_model_size(self, model):
        model_dtype = np.dtype(model.dtype)
        variables = int(np.sum([np.prod(v.shape) for v in model.variables]))
        size = variables * model_dtype.itemsize
        return size, model_dtype

    def get_shared_memory_reference(self, model, memory_name):
        ## return the reference to the memory and the np.array pointing to the shared memory
        size, model_dtype = self.compute_model_size(model)
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initialize_models(self):
        try:
            print('Creating neural networks...', end='')
            stage = 'Creating the neural networks'
            self.ddpg_agent = DDPGAgent(self.tf, self.context_size, self.action_size)
            self.ddpg_agent.load_actor()
            self.ddpg_agent.load_critic()
            print('Done')

            print('Actor memory reference creation...', end='')
            stage = 'Actor memory reference creation'
            self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.ddpg_agent.actor, self.actor_memory_name)
            self.weights_actor = map_weights_to_shared_memory_buffer(self.ddpg_agent.actor.get_weights(), self.np_array_actor)
            print('Done')

            print('Critic memory reference creation...', end='')
            stage = 'Critic memory reference creation'
            self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.ddpg_agent.critic, self.critic_memory_name)
            self.weights_critic = map_weights_to_shared_memory_buffer(self.ddpg_agent.critic.get_weights(), self.np_array_critic)
            print('Done')
        except Exception as e:
            print(str(self) + ' -> Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def load_weights(self):
         if (self.load_initial_weights):
                print(str(self) + ' -> Loading actor initial weights from ' + self.actor_initial_weights_path)
                self.ddpg_agent.load_actor_weights(self.actor_initial_weights_path)
                print(str(self) + ' -> Loading critic initial weights from ' + self.critic_initial_weights_path)
                self.ddpg_agent.load_critic_weights(self.critic_initial_weights_path)

    def publish_weights(self):
        print(str(self) + ' -> Publishing actor weights to shared memory...', end='')
        publish_weights_to_shared_memory(self.ddpg_agent.actor.get_weights(), self.np_array_actor)                
        print('Done')
        
        print(str(self) + ' -> Publishing critic weights to shared memory', end='')   
        publish_weights_to_shared_memory(self.ddpg_agent.critic.get_weights(), self.np_array_critic)
        print('Done')

    def __str__(self) -> str:
        return 'Main Agent'

    

        
        


