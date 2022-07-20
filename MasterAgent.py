import multiprocessing as mp
from BaseAgent import BaseAgent

from common_utils import get_basic_actor_network, get_basic_critic_network, get_shared_memory_ref, import_tensorflow, map_weights_to_shared_memory_buffer, publish_weights_to_shared_memory, save_weights
import random
import numpy as np

class Master_Agent(mp.Process):
    def __init__(self,
                hyperparameters,
                config, state_size, action_size,
                actor_memory_bytes: mp.Value, critic_memory_bytes: mp.Value,
                memory_created: mp.Value, master_agent_initialized: mp.Value,
                gradient_updates_queue: mp.Queue, master_agent_stop: mp.Value,
                optimizer_lock: mp.Lock,
                actor_memory_name = 'model_actor',
                critic_memory_name = 'model_critic') -> None:
        super(Master_Agent, self).__init__()
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.hyperparameters = hyperparameters
        self.actor_memory_bytes = actor_memory_bytes
        self.critic_memory_bytes = critic_memory_bytes
        self.memory_created = memory_created
        self.master_agent_initialized = master_agent_initialized
        self.gradient_updates_queue = gradient_updates_queue
        self.optimizer_lock = optimizer_lock
        self.master_agent_stop = master_agent_stop
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name

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

    def save_weights(self):
        save_weights(self.actor, self.config.save_weights_file, False)
        save_weights(self.critic, self.config.save_weights_file + '_critic.h5')

    def compute_gradients(self, a2c_ac_grads, a2c_cr_grads, sil_ac_grads, sil_cr_grads):
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

    def run(self):
        self.tf, _, self.tfp = import_tensorflow('3', True)
        self.set_process_seeds()  
        exited_successfully = False
        try:
            self.initiate_models()            
            self.tf_compute_gradients = self.tf.function(self.compute_gradients)

            if (self.config.load_initial_weights):
                print(str(self) + ' -> Loading initial weights from ' + self.config.initial_weights_path)
                self.actor.load_weights(self.config.initial_weights_path)
            
            publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)                
            print(str(self) + ' -> Published actor weights to shared memory')
            
            publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)
            print(str(self) + ' -> Published critic weights to shared memory')
            self.master_agent_initialized.value = 1

            self.actor_critic_optimizer = self.tf.keras.optimizers.Adam(learning_rate = self.hyperparameters['Actor_Critic_Common']['learning_rate'])
            gradient_calculation_idx = 0
            while True:
                import queue
                try:
                    gradients = self.gradient_updates_queue.get(block = True, timeout = 10)
                    self.tf_compute_gradients(gradients[0], gradients[1], gradients[2], gradients[3])

                    with self.optimizer_lock:
                        print(str(self) + ' -> Pushing new weights...')
                        publish_weights_to_shared_memory(self.actor.get_weights(), self.np_array_actor)
                        publish_weights_to_shared_memory(self.critic.get_weights(), self.np_array_critic)

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
        


