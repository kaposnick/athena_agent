import multiprocessing as mp
import numpy as np
import random
from OUActionNoise import OUActionNoise
from Buffer import EpisodeBuffer
from env.BaseEnv import BaseEnv
from common_utils import get_basic_actor_network, get_basic_critic_network, import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer
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
                in_scheduling_mode: str,
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
        self.in_scheduling_mode = in_scheduling_mode
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name

        self.in_training_mode = in_training_mode
        self.worker_agent_stop_value = worker_agent_stop_value

        self.init_configuration()        

    def __str__(self) -> str:
        return 'Worker ' + str(self.worker_num)
        
    def init_configuration(self):
        self.local_update_period = self.config.hyperparameters['Actor_Critic_Common']['local_update_period']
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']
        

    

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
        self.buffer = EpisodeBuffer(3000, self.batch_size, self.state_size, 1)
        self.coef = 5.159817058590249
        self.intercept = 7.6586417701293446


    def update_weights(self):
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
            self.send_results(self.batch_info)

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
        state_batch, action_batch, reward_batch = self.buffer.sample(self.tf)
        return self.tf_compute_grads(state_batch, action_batch, reward_batch, add_entropy_term = False)

    def compute_a2c_grads(self):
        state_batch, action_batch, reward_batch = self.buffer.sample_sil(self.tf, self.critic)
        return self.tf_compute_grads(state_batch, action_batch, reward_batch, add_entropy_term = True)


    def learn(self):
        a2c_actor_grads, a2c_critic_grads, a2c_info = self.compute_a2c_grads()
        sil_actor_grads, sil_critic_grads, sil_info = self.compute_sil_grads()

        gradients = [a2c_actor_grads, a2c_critic_grads, sil_actor_grads, sil_critic_grads]
        self.gradient_updates_queue.put(gradients)
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

    def convert_to_real_action_applied(self, info):
        mcs = int(info['mcs'])
        prb = int(info['prb'])
        tbs = self.environment.to_tbs(mcs, prb)
        return (np.log(tbs + 1) - self.intercept) / self.coef

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
            self.tf_compute_grads = self.tf.function(self.compute_grads)
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            self.ep_ix = 0
            while (True):
                self.ep_ix += 1
                if (self.ep_ix % 1 == 0):
                    self.print('Episode {}'.format(self.ep_ix + 1))
                if (self.in_training_mode.value == 1 and (self.ep_ix % self.local_update_period == 0)):
                    update_weights_time = time.time()
                    self.update_weights()
                    self.print('Updating weights time: {}'.format(time.time() - update_weights_time))

                self.batch_info = []
                self.sample_buffer = []

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
                        if (reward is None):
                            self.print('Action not applied.. skipping')
                            state = next_state
                            continue
                        self.print('Executing action time: {}'.format(time.time() - step_time))

                        real_action_applied = self.convert_to_real_action_applied(info)

                        if (self.in_training_mode.value == 1):                        
                            self.sample_buffer.append((state, real_action_applied, reward))
                        
                        self.batch_info.append(info)
                        state = next_state
                        batch_idx += 1

                if (self.in_training_mode.value == 1):
                    self.sample_buffer_queue.put(self.sample_buffer)
                self.batch_info_queue.put(self.batch_info)   
                
        finally:
            print(str(self) + ' -> Exiting...')

    def pick_action_from_embedding_table(self, state: np.array):
        in_training_mode = self.in_training_mode.value
        actor_input = self.tf.convert_to_tensor([state], dtype = self.tf.float32)
        distr = self.actor(actor_input)

        if (in_training_mode):
            heta_param = np.random.normal(loc = 0, scale = 1)
            mean = distr.mean()
            stddev = distr.stddev()
            action_hat = mean + heta_param * stddev
        else:
            action_hat = distr.mean()
        # action_hat = mean

        tbs_hat = np.exp(self.coef * action_hat + self.intercept) - 1
        
        action_idx, tbs = self.environment.get_closest_actions(tbs_hat)

        action = (np.log(tbs + 1) - self.intercept) / self.coef
        self.print('tbs hat: {} - tbs: {}'.format(tbs_hat, tbs))
        if (in_training_mode):
            self.print('action hat: {}/{}/{} - action: {}'.format(mean[0][0].numpy(), stddev[0][0].numpy(), action_hat, action))
        return action_idx, action

    def print(self, string_to_print, end = None):
        if (self.worker_num > 10):
            print(str(self) + ' -> ' + string_to_print, end=end)



