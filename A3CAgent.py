import copy
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import random
import gym
import numpy as np
import BaseAgent
import multiprocessing as mp
from BaseAgent import BaseAgent
from common_utils import import_tensorflow
from env.SrsRanEnv import MCS_SPACE, PRB_SPACE

COL_DT_AC_LOSS_MEAN = 'ac_loss_mean'
COL_DT_CR_LOSS_MEAN = 'cr_loss_mean'
COL_DT_ENTROPY_MEAN = 'entropy'
COL_DT_MCS_MEAN = 'mcs_mean'
COL_DT_PRB_MEAN = 'prb_mean'
COL_DT_REP = 'rep'
COL_DT_REWARD_MEAN = 'reward_mean'
COL_DT_SIM = 'sim'

class A3CAgent(BaseAgent):
    agent_name = "A3C"
    def __init__(self, config, num_processes) -> None:
        super(A3CAgent, self).__init__(config)  
        self.num_processes = num_processes

    def run_n_episodes(self, processes_started_successfully = None):
        results_queue = mp.Queue()
        gradient_updates_queue = mp.Queue()
        results_queue = mp.Queue()
        
        episode_number = mp.Value('i', 0)
        
        memory_size_in_bytes = mp.Value('i', 0)
        memory_created = mp.Value('i')
        global_ac_initialized = mp.Value('i', 0)
        global_process_stop = mp.Value('i', 0)
        
        self.optimizer_lock = mp.Lock()
        save_file = self.config.results_file_path
        self.write_to_results_queue_lock = mp.Lock()

        if (self.config.num_episodes_to_run > 0):
            episodes_per_process = int(self.config.num_episodes_to_run / self.num_processes)
        else:
            episodes_per_process = -1 # run indefinitely

        processes = []

        optimizer_worker = mp.Process(target = self.update_shared_model, 
                                    args = (gradient_updates_queue,
                                            self.hyperparameters,
                                            memory_size_in_bytes, 
                                            memory_created, global_ac_initialized,
                                            global_process_stop))

        
        import signal
        signal.signal(signal.SIGINT , self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        optimizer_worker.start()
        while (memory_size_in_bytes.value == 0):
            pass
        size_in_bytes = memory_size_in_bytes.value

        try:
            try:
                self.shm = shared_memory.SharedMemory(name = 'model_weights_12132', create = True, size=size_in_bytes)
            except:
                self.shm = shared_memory.SharedMemory(name = 'model_weights_12132', create = False, size=size_in_bytes)
            memory_created.value = 1
                        
            while (global_ac_initialized.value == 0):
                pass
            
            print('Optimizer thread started successfully')
            successfully_started_worker = mp.Value('i', 0)
            for process_num in range(self.num_processes):
                worker_environment = copy.deepcopy(self.environment)
                worker = Actor_Critic_Worker(process_num, 
                                            self.num_processes,
                                            successfully_started_worker,
                                            worker_environment, self.optimizer_lock, self.write_to_results_queue_lock,
                                            self.config, episodes_per_process, 
                                            self.state_size, self.action_size, self.action_types, 
                                            results_queue, gradient_updates_queue, episode_number)
                worker.start()
                processes.append(worker)
            while (successfully_started_worker.value < self.num_processes):
                pass

            if (processes_started_successfully is not None):
                processes_started_successfully.value = 1
            if (self.config.save_results):
                self.save_results(save_file, episode_number, results_queue)
            for worker in processes:
                worker.join()
            global_process_stop.value = 1
            optimizer_worker.join()
        finally:
            self.exit_gracefully(True)


    def save_results(self, save_file, episode_number, results_queue):
        with open(save_file, 'w') as f:
            additional_env_columns = self.environment.get_csv_result_policy_output_columns()            
            add_columns_size = len(additional_env_columns)
            f.write('|'.join(COLUMNS + additional_env_columns) + '\n')            
            while True:
                with self.write_to_results_queue_lock:
                    carry_on = episode_number.value < self.config.num_episodes_to_run
                    episode = episode_number.value
                if carry_on:
                    if not results_queue.empty():
                        q_result = results_queue.get()
                        result = [str(x) for x in [episode, *q_result[0]]]
                        if (episode % 100 == 0):
                            additional_info = [str(x).replace('\n', '') for x in q_result[1]]
                        else:
                            additional_info = [''] * add_columns_size 
                        result += additional_info
                        f.write('|'.join(result) + '\n')
                        f.flush()
                else: break


    @staticmethod
    def get_shared_memory_ref(
        size, dtype, share_memory_name = 'model_weights_12132'):
        total_variables = int( size / dtype.itemsize )
        shm = shared_memory.SharedMemory(name=share_memory_name, create=False, size=size)        
        shared_weights_array = np.ndarray(
                                shape = (total_variables, ),
                                dtype = dtype,
                                buffer = shm.buf)
        return shm, shared_weights_array

    @staticmethod
    def publish_weights_to_shared_memory(weights, shared_ndarray):
        buffer_idx = 0
        for weight in weights:
            flattened = weight.flatten().tolist()
            size = len(flattened)
            shared_ndarray[buffer_idx: (buffer_idx + size)] = flattened
            buffer_idx += size

    @staticmethod
    def map_weights_to_shared_memory(weights, shared_weights_array):
        buffer_idx = 0
        for idx_weight in range(len(weights)):
            weight_i = weights[idx_weight]
            shape = weight_i.shape
            size  = weight_i.size
            weights[idx_weight] = np.ndarray(shape = shape,
                                           dtype = weight_i.dtype,
                                           buffer = shared_weights_array[buffer_idx: (buffer_idx + size)])
            buffer_idx += size

        return weights

    def exit_gracefully(self, unlink = True):
        if hasattr(self, 'shared_weights_array'):
            try:
                if (self.shared_weights_array is not None):
                    del self.shared_weights_array
            finally: 
                pass
        if hasattr(self, 'shm'):
            try:
                if (self.shm != None):                        
                    self.shm.close()

                    if (unlink):
                        self.shm.unlink()
            except:
                pass
            finally:
                pass

    
    def update_shared_model(self, gradient_updates_queue, hyperparameters,
                    memory_size_in_bytes, 
                    memory_created, global_ac_initialized,
                    global_process_stop):
        import signal
        signal.signal(signal.SIGINT , self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        exited_successfully = False
        try:
            learning_rate = hyperparameters['Actor_Critic_Common']['learning_rate']
            tf, os = import_tensorflow('3')
            import queue
            os.environ['PYTHONHASHSEED'] = str(self.config.seed)
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            tf.random.set_seed(self.config.seed)  
            self.global_model = BaseAgent.create_NN(tf, 
                                self.state_size, 
                                [*self.action_size, 1], 
                                self.config.hyperparameters)
            total_variables = np.sum([np.prod(v.shape) for v in self.global_model.trainable_variables])
            model_dtype = np.dtype(self.global_model.dtype)
            memory_size_in_bytes.value = total_variables * model_dtype.itemsize
            
            while (memory_created.value == 0):
                pass

            self.shm, self.shared_weights_array = A3CAgent.get_shared_memory_ref(
                memory_size_in_bytes.value,
                model_dtype)
            if (self.config.load_initial_weights):
                print('Loading previous weights to continue training')
                self.global_model.load_weights(self.config.initial_weights_path)
            else:
                print('Not loading any weights')
            A3CAgent.publish_weights_to_shared_memory(self.global_model.get_weights(), self.shared_weights_array)                
            print('Global Agent published weights')
            global_ac_initialized.value = 1

            actor_critic_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            gradient_calculation_idx = 0
            while True:
                try:
                    gradients = gradient_updates_queue.get(block = True, timeout = 10)
                    clipped_gradients = [(tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1)) for grad in gradients]
                    actor_critic_optimizer.apply_gradients(
                        zip(clipped_gradients, self.global_model.trainable_weights)
                    )

                    for weight in self.global_model.get_weights():
                        if (np.isnan(weight).any()):
                            print('Global agent has weights with NaN elements')
                            self.save_weights(self.config.save_weights_file)
                            break

                    with self.optimizer_lock:
                        A3CAgent.publish_weights_to_shared_memory(self.global_model.get_weights(), self.shared_weights_array)

                    gradient_calculation_idx += 1
                    if (self.config.save_weights and gradient_calculation_idx % self.config.save_weights_period == 0):
                        self.save_weights(self.config.save_weights_file)
                except queue.Empty:
                    if (global_process_stop.value == 1):
                        exited_successfully = True
                        break
        finally:
            print('Exiting gracefully...')
            if (exited_successfully):
                print('Exiting successfully after all episodes run...')
                if (self.config.save_weights):
                    self.save_weights(self.config.save_weights_file)
            else:                
                self.save_weights('/home/naposto/phd/nokia/data/csv_45/entropy_0.1_model_error_happened.h5')
            self.exit_gracefully(False)
    
    def save_weights(self, save_weights_file):
        try:
            print('Saving weights...', end='')
            self.global_model.save_weights(save_weights_file)
            print('Success')
        except Exception as e:
            print('Error' + str(e))

            

COLUMNS = [
    COL_DT_REP, COL_DT_SIM,
    COL_DT_REWARD_MEAN, COL_DT_ENTROPY_MEAN, COL_DT_MCS_MEAN, COL_DT_PRB_MEAN, COL_DT_AC_LOSS_MEAN, COL_DT_CR_LOSS_MEAN
] 

class Actor_Critic_Worker(mp.Process):
    def __init__(self, worker_num, total_workers, successfully_started_worker, environment, optimizer_lock, write_to_results_queue_lock,
                 config, episodes_to_run, state_size, action_size, action_types, results_queue, gradient_updates_queue,
                 episode_number) -> None:
        super(Actor_Critic_Worker, self).__init__()
        self.environment = environment
        self.config = config
        self.worker_num = worker_num
        self.total_workers = total_workers
        self.successfully_started_worker = successfully_started_worker

        self.state_size = state_size
        self.action_size = action_size
        self.optimizer_lock = optimizer_lock
        self.write_to_results_queue_lock = write_to_results_queue_lock
        self.episodes_to_run = episodes_to_run
        self.action_types = action_types
        self.results_queue = results_queue
        self.counter = episode_number
        self.results_queue = results_queue

        self.gradient_updates_queue = gradient_updates_queue
        self.local_update_period = self.config.hyperparameters['Actor_Critic_Common']['local_update_period']
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']
        self.include_entropy_term = self.config.hyperparameters['Actor_Critic_Common']['include_entropy_term']
        if (self.include_entropy_term):            
            self.entropy_beta = self.config.hyperparameters['Actor_Critic_Common']['entropy_beta']
            self.entropy_contrib_prob = self.config.hyperparameters['Actor_Critic_Common']['entropy_contrib_prob']
        

    def set_process_seeds(self, tf, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed + worker_num)
        random.seed(self.config.seed + worker_num)
        np.random.seed(self.config.seed + worker_num)
        tf.random.set_seed(self.config.seed + worker_num)

    def run(self) -> None:
        tf, _ = import_tensorflow('3')
        import copy 
        self.set_process_seeds(tf, self.worker_num)
        self.local_model = BaseAgent.create_NN(tf, 
                            self.state_size, 
                            [*self.action_size, 1], 
                            self.config.hyperparameters)
        try:
            model_dtype = np.dtype(self.local_model.dtype)
            total_variables = np.sum([np.prod(v.shape) for v in self.local_model.trainable_variables])
            size = total_variables * model_dtype.itemsize

            self.shm, self.shared_weights_array = A3CAgent.get_shared_memory_ref(
                size, model_dtype)
            self.global_weights = A3CAgent.map_weights_to_shared_memory(
                                self.local_model.get_weights(), self.shared_weights_array)

            self.environment.setup(self.worker_num, self.total_workers)
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
                
            
            num_outputs = len(self.action_size)
            for ep_ix in range(self.episodes_to_run):
                if (ep_ix % 1 == 0):
                    print('Episode {}/{}'.format(ep_ix, self.episodes_to_run))
                if (ep_ix % self.local_update_period == 0):
                    with self.optimizer_lock:
                        self.local_model.set_weights(copy.deepcopy(self.global_weights))

                self.batch_action = []
                self.batch_logp = []
                self.batch_rewards = []
                self.batch_critic_outputs = []
                self.batch_info = []

                self.times_ac = []
                self.times_env = []

                with tf.GradientTape() as tape:
                    batch_idx = 0
                    while batch_idx < self.batch_size:
                        state = self.reset_game_for_worker()
                        done = False
                        while not done:
                            action, logp, critic_output = self.pick_action_and_get_critic_values(self.local_model, state, tf)
                            next_state, reward, done, info = self.environment.step(action)
                            
                            sample_action_entry = []
                            sample_logp_entry   = []
                            for output_idx in range(num_outputs):
                                sample_logp_entry.append(logp[output_idx])
                                sample_action_entry.append(action[output_idx])
                            self.batch_action.append(sample_action_entry)
                            self.batch_logp.append(sample_logp_entry)
                            self.batch_rewards.append(reward)
                            self.batch_critic_outputs.append(critic_output)
                            self.batch_info.append(info)
                            state = next_state
                            batch_idx += 1
                    
                    advantage = tf.expand_dims(tf.convert_to_tensor(self.batch_rewards, dtype = tf.float32), axis = 1) -  \
                                tf.squeeze(tf.convert_to_tensor(self.batch_critic_outputs, dtype = tf.float32), axis = 2)
                    critic_loss = advantage ** 2

                    tensor_batch_action = [tf.convert_to_tensor([row[idx] for row in self.batch_action]) for idx in range(num_outputs)]
                    tensor_batch_logp   = [tf.convert_to_tensor([row[idx] for row in self.batch_logp]) for idx in range(num_outputs)]
                    tensor_batch_p      = [tf.math.exp(tensor_batch_logp[idx]) for idx in range(num_outputs)]

                    joint_action_logp = []
                    for batch_idx in range(self.batch_size):
                        sum = []
                        for action_idx in range(num_outputs):
                            sum.append(tensor_batch_logp[action_idx][batch_idx][0][ tensor_batch_action[action_idx][batch_idx] ])
                        joint_action_logp.append(tf.reduce_sum(sum))
                    joint_action_logp = tf.expand_dims(joint_action_logp, axis = 1)

                    actor_loss_inside_term = joint_action_logp * advantage # [batch_size x 1]
                    
                    if (self.include_entropy_term):
                        joint_probs = []
                        for batch_idx in range(self.batch_size):
                            entry = tensor_batch_p[0][batch_idx]
                            for action_idx in range(1, num_outputs):
                                entry_2 = tensor_batch_p[action_idx][batch_idx]
                                entry = tf.linalg.matmul(entry, entry_2, transpose_a = True, name = 'joint_prob')
                                entry = tf.expand_dims(tf.reshape(entry, [-1]), axis = 0)
                            joint_probs.append(entry)

                        joint_probs = tf.convert_to_tensor(joint_probs)                       
                        
                        plogp = tf.math.xlogy(joint_probs, joint_probs, name = 'plogp')
                        entropy = -1 * tf.expand_dims(tf.reduce_sum(plogp, axis = [1, 2], name = 'batch_entropy'), axis = 1)
                        
                        entropy_contribution = np.random.binomial(1, np.power(self.entropy_contrib_prob, ep_ix))
                        actor_loss_inside_term += entropy_contribution * self.entropy_beta * entropy

                    actor_loss = -1 * actor_loss_inside_term

                    actor_loss_mean = tf.reduce_mean(actor_loss)
                    critic_loss_mean = tf.reduce_mean(critic_loss)
                    
                    total_loss = actor_loss_mean + critic_loss_mean                    
                gradients = tape.gradient(total_loss, self.local_model.trainable_weights)

                for weight in self.local_model.get_weights():
                        if (np.isnan(weight).any()):
                            print('Local agent {} has weights with NaN elements'.format(self.worker_num))
                            break

                for gradient in gradients:
                        if (np.isnan(gradient).any()):
                            print('Local agent {} has gradient with NaN elements'.format(self.worker_num))
                            break


                self.gradient_updates_queue.put(gradients)

                probs_batched = [tf.reduce_mean(tensor_batch_p[action_idx], axis = 0)   for action_idx in range(num_outputs)]
                
                rew_mean = np.mean(self.batch_rewards)
                entropy_mean = np.mean(entropy)
                mcs_mean, prb_mean = self.environment.calculate_mean(probs_batched)
                additional_columns = self.environment.get_csv_result_policy_output(probs_batched)

                with self.write_to_results_queue_lock:
                    self.counter.value += 1
                    self.results_queue.put([ [self.worker_num, rew_mean, entropy_mean, mcs_mean, prb_mean, actor_loss_mean.numpy(), critic_loss_mean.numpy()], additional_columns ])
                
        finally:
            if hasattr(self, 'shared_weights_array'):
                try:
                    if (self.shared_weights_array is not None):
                        del self.shared_weights_array
                finally: 
                    pass
            if hasattr(self, 'shm'):
                try:
                    if (self.shm != None):                        
                        self.shm.close()
                except:
                    pass
                finally:
                    pass
        

    def reset_game_for_worker(self):
        state = self.environment.reset()
        return state

    def pick_action_and_get_critic_values(self, policy, state, tf):
        state = tf.convert_to_tensor([state], dtype = tf.float32)
        
        model_output = policy(state)
        actor_output_probs = model_output[:len(self.action_size)] # normalized probs
        critic_output = model_output[-1]

        actor_output_log_probs = [tf.math.log(output + 1e-10) for output in actor_output_probs]
        actor_samples = [tf.random.categorical(output, num_samples = 1) for output in actor_output_log_probs]
        action = [actor_sample[0][0].numpy() for actor_sample in actor_samples]
        return action, actor_output_log_probs, critic_output   


