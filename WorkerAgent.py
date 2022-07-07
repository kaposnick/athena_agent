import multiprocessing as mp
import numpy as np
import random
from env.BaseEnv import BaseEnv
from common_utils import import_tensorflow, get_shared_memory_ref, map_weights_to_shared_memory_buffer
from BaseAgent import BaseAgent
from Config import Config

class Actor_Critic_Worker(mp.Process):
    def __init__(self, 
                environment: BaseEnv, config: Config,
                worker_num: np.int32, total_workers: np.int32, 
                episodes_to_run: np.int32, state_size, action_size, 
                successfully_started_worker: mp.Value, 
                optimizer_lock: mp.Lock, write_to_results_queue_lock: mp.Lock,
                results_queue: mp.Queue, gradient_updates_queue: mp.Queue,
                episode_number: mp.Value,
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
        self.results_queue = results_queue
        self.gradient_updates_queue = gradient_updates_queue
        self.episode_number = episode_number
        self.actor_memory_name = actor_memory_name
        self.critic_memory_name = critic_memory_name

        self.init_configuration()        

    def __str__(self) -> str:
        return 'Worker ' + str(self.worker_num)
        
    def init_configuration(self):
        self.local_update_period = self.config.hyperparameters['Actor_Critic_Common']['local_update_period']
        self.batch_size = self.config.hyperparameters['Actor_Critic_Common']['batch_size']
        self.include_entropy_term = self.config.hyperparameters['Actor_Critic_Common']['include_entropy_term']
        self.use_action_value_critic = not self.config.hyperparameters['Actor_Critic_Common']['use_state_value_critic']
        if (self.use_action_value_critic):
            self.vmin    = self.config.hyperparameters['Actor_Critic_Common']['Action_Value_Critic']['vmin']
            self.vmax    = self.config.hyperparameters['Actor_Critic_Common']['Action_Value_Critic']['vmax']
            self.n_atoms = self.config.hyperparameters['Actor_Critic_Common']['Action_Value_Critic']['n_atoms']
            self.delta = (self.vmax - self.vmin) / self.n_atoms
            
        if (self.include_entropy_term):            
            self.entropy_beta = self.config.hyperparameters['Actor_Critic_Common']['entropy_beta']
            self.entropy_contrib_prob = self.config.hyperparameters['Actor_Critic_Common']['entropy_contrib_prob']

    

    def set_process_seeds(self, tf, worker_num):
        import os
        os.environ['PYTHONHASHSEED'] = str(self.config.seed + worker_num)
        random.seed(self.config.seed + worker_num)
        np.random.seed(self.config.seed + worker_num)
        tf.random.set_seed(self.config.seed + worker_num)

    def get_shared_memory_reference(self, model, memory_name):
        ## return the reference to the memory and the np.array pointing to the shared memory
        model_dtype = np.dtype(model.dtype)
        variables = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        size = variables * model_dtype.itemsize
        shm, weights_array = get_shared_memory_ref(size, model_dtype, memory_name)
        return shm, weights_array

    def initiate_models(self, tf):
        try:
            stage = 'Creating the neural networks'
            models = BaseAgent.create_NN(tf, 
                                self.state_size, 
                                [*self.action_size, 1], 
                                self.config.hyperparameters)

            stage = 'Actor memory reference creation'
            self.actor = models[0]
            self.shm_actor, self.np_array_actor = self.get_shared_memory_reference(self.actor, self.actor_memory_name)
            self.weights_actor = map_weights_to_shared_memory_buffer(self.actor.get_weights(), self.np_array_actor)

            if (self.use_action_value_critic):
                stage = 'Action-value critic memory reference creation'
                self.critic = models[1]
                self.shm_critic, self.np_array_critic = self.get_shared_memory_reference(self.critic, self.critic_memory_name)
                self.weights_critic = map_weights_to_shared_memory_buffer(self.critic.get_weights(), self.np_array_critic)
        except Exception as e:
            print(str(self) + ' -> Stage: {}, Error initiating models: {}'.format(stage, e))
            raise e

    def initiate_worker_variables(self, tf):
        self.batch_range = tf.expand_dims(tf.range(0, self.batch_size), axis = 1)
        if (self.use_action_value_critic):
            self.atoms_range = tf.expand_dims(tf.range(start = self.vmin, limit = self.vmax, delta = self.delta, dtype = tf.float32), axis = 1)

    def update_weights(self):
        import copy
        with self.optimizer_lock:
            self.actor.set_weights(copy.deepcopy(self.weights_actor))
            if (self.use_action_value_critic):
                self.critic.set_weights(copy.deepcopy(self.weights_critic))

    def compute_critic_loss(self, tf, rewards, z, actions):
        atoms = rewards
        b = (atoms - self.vmin) / self.delta
        l = tf.math.floor(b)
        u = tf.clip_by_value(tf.math.ceil(b), 
                            clip_value_min = 0,
                            clip_value_max = (self.n_atoms - 1))
        probabilities = tf.ones((self.batch_size, self.n_atoms)) / self.n_atoms
        lower = probabilities * (u - b)
        upper = probabilities * (b - l)
        z_projected = tf.zeros_like(probabilities)

        l_cast = tf.cast(l, dtype = tf.int32)
        u_cast = tf.cast(u, dtype = tf.int32)
        z_projected = tf.tensor_scatter_nd_add(tensor  = z_projected, 
                                            indices = tf.concat([self.batch_range, l_cast], axis = 1), 
                                            updates = tf.squeeze(tf.gather(lower, l_cast, batch_dims = 1)))
        

        z_projected = tf.tensor_scatter_nd_add(tensor = z_projected, 
                                            indices = tf.concat([self.batch_range, u_cast], axis = 1), 
                                            updates = tf.squeeze(tf.gather(upper, u_cast, batch_dims = 1)))
        z_stacked = tf.stack([sample_z[sample_action[0]] for sample_z, sample_action in zip(z, actions)])
        critic_loss = -1 * tf.reduce_mean(tf.reduce_sum((z_projected * tf.math.log(z_stacked)), axis = -1))
        return critic_loss
        

    def compute_actor_loss(self, tf, rewards, z):
        """
            rewards: tf.Tensor with shape (batch_size, 1)
            z:       tf.Tensor with shape (batch_size, n_actions, n_atoms)
        """
        mean_critic_value = tf.linalg.matmul(z, self.atoms_range) # (batch_size, n_actions, 1)
        mean_critic_value = tf.reduce_mean(mean_critic_value, axis = [1])
        advantage = rewards - mean_critic_value

        tensor_batch_action = tf.convert_to_tensor(self.batch_action) 
        tensor_batch_logp   = tf.squeeze(tf.convert_to_tensor(self.batch_logp), axis = 1)
        tensor_batch_p      = tf.math.exp(tensor_batch_logp)

        joint_action_logp = []
        for batch_idx in range(self.batch_size):
            joint_action_logp.append(tensor_batch_logp[batch_idx][tensor_batch_action[batch_idx][0].numpy()])
        joint_action_logp = tf.convert_to_tensor(joint_action_logp)
        joint_action_logp = tf.expand_dims(joint_action_logp, axis = 1)

        actor_loss_inside_term = joint_action_logp * advantage # [batch_size x 1]
        
        entropy = -1 * tf.expand_dims(
            tf.reduce_sum(tf.math.xlogy(tensor_batch_p, tensor_batch_p), axis = 1),
            axis = 1
        )                        
        if (self.include_entropy_term):
            entropy_contribution = np.random.binomial(1, np.power(self.entropy_contrib_prob, self.ep_ix))
            actor_loss_inside_term += entropy_contribution * self.entropy_beta * entropy
        actor_loss = -1 * tf.reduce_mean(actor_loss_inside_term)
        info_entropy = np.mean(entropy)

        info = {
            'tensor_batch_p': tensor_batch_p, 
            'entropy':        info_entropy
        }
        
        return actor_loss, info

    def compute_losses(self, tf):
        rewards = tf.expand_dims(tf.convert_to_tensor(self.batch_rewards, dtype = tf.float32), axis = 1) # (batch_size, 1)
        actions = tf.convert_to_tensor(self.batch_action)
        z = tf.squeeze(tf.convert_to_tensor(self.batch_critic_outputs, dtype = tf.float32))              # (batch_size, actions, atoms)

        actor_loss, actor_info  = self.compute_actor_loss(tf, rewards, z)
        critic_loss = None
        if (self.use_action_value_critic):
            critic_loss = self.compute_critic_loss(tf, rewards, z, actions)
        
        ret_info = {
            'tensor_batch_p': actor_info['tensor_batch_p'],
            'entropy'       : actor_info['entropy'],
            'actor_loss'    : actor_loss.numpy(),
            'critic_loss'   : critic_loss.numpy(),
        }
        return actor_loss, critic_loss, ret_info

    def send_gradients(self, tape, actor_loss, critic_loss):
        gradients = [tape.gradient(actor_loss, self.actor.trainable_weights)]
        if (self.use_action_value_critic):
            gradients.append(tape.gradient(critic_loss, self.critic.trainable_weights))
        self.gradient_updates_queue.put(gradients)
        return gradients

    def send_results(self, tf, info):
        tensor_batch_p = info['tensor_batch_p']
        entropy        = info['entropy']
        actor_loss     = info['actor_loss']
        critic_loss    = info['critic_loss']
        probs_batched = [tf.reduce_mean(tensor_batch_p, axis = 0)]
                
        rew_mean = np.mean(self.batch_rewards)
        mcs_mean, prb_mean = self.environment.calculate_mean(probs_batched)
        additional_columns = self.environment.get_csv_result_policy_output(probs_batched, self.batch_info)

        with self.write_to_results_queue_lock:
            self.episode_number.value += 1
            self.results_queue.put([ [self.worker_num, rew_mean, entropy, mcs_mean, prb_mean, actor_loss, critic_loss], additional_columns ])


    def run(self) -> None:
        tf, _ = import_tensorflow('3')
        self.set_process_seeds(tf, self.worker_num)        
        self.initiate_worker_variables(tf)        
        try:
            self.initiate_models(tf)
            self.environment.setup(self.worker_num, self.total_workers)
            
            with self.successfully_started_worker.get_lock():
                self.successfully_started_worker.value += 1
            
            for self.ep_ix in range(self.episodes_to_run):
                if (self.ep_ix % 1 == 0):
                    print(str(self) + ' -> Episode {}/{}'.format(self.ep_ix + 1, self.episodes_to_run))
                if (self.ep_ix % self.local_update_period == 0):
                    self.update_weights()

                self.batch_action = []
                self.batch_logp = []
                self.batch_rewards = []
                self.batch_critic_outputs = []
                self.batch_info = []


                with tf.GradientTape(persistent=True) as tape:
                    batch_idx = 0
                    while batch_idx < self.batch_size:
                        state = self.environment.reset()
                        done = False
                        while not done:
                            action, logp, critic_output = self.pick_action_and_get_critic_values(state, tf)
                            next_state, reward, done, info = self.environment.step(action)
                            
                            self.batch_action.append(action)
                            self.batch_logp.append(logp)
                            self.batch_rewards.append(reward)
                            self.batch_critic_outputs.append(critic_output)
                            self.batch_info.append(info)
                            state = next_state
                            batch_idx += 1
                    
                    actor_loss, critic_loss, info = self.compute_losses(tf)
                    
                self.send_gradients(tape, actor_loss, critic_loss)
                del tape
                
                self.send_results(tf, info)                
                
        finally:
            print(str(self) + ' -> Exiting...')

    def pick_action_and_get_critic_values(self, state: np.array, tf):
        tensor_state = tf.convert_to_tensor([state], dtype = tf.float32)
        
        model_output = self.actor(tensor_state)
        actor_output_probs = model_output

        actor_output_log_probs = tf.math.log(actor_output_probs + 1e-10)
        actor_sample = tf.random.categorical(actor_output_log_probs, num_samples = 1) 
        action = [actor_sample[0][0].numpy()]
        if (self.use_action_value_critic):
            tensor_state = tf.convert_to_tensor([state])
            critic_output = self.critic(tensor_state) # n_actions * n_atoms
        return action, actor_output_log_probs, critic_output   


