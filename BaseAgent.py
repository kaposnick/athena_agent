import logging
import os
import gym
import numpy as np

import time

from env.DecoderEnv import BETA_MAX, BETA_MIN, BSR_MAX, BSR_MIN, NOISE_MAX, NOISE_MIN

class BaseAgent(object):
    def __init__(self, config) -> None:
        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        self.config = config
        self.action_types = "DISCRETE"
        self.environment = config.environment
        
        self.action_size = self.get_action_size()
        self.config.action_size = self.action_size
        
        self.state_size = self.environment.get_state_size()
        self.config.state_size = self.state_size
        
        self.episode_number = 0
        self.global_step_number = 0
        self.hyperparameters = config.hyperparameters

    @staticmethod
    def create_NN(tf, input_dim, output_dim, hyperparameters, name = 'base_agent_model'):
        # hyperparameters = self.hyperparameters
        layers = tf.keras.layers

        ac_common_params = hyperparameters['Actor_Critic_Common']
        ac_common_hidden_units = ac_common_params['linear_hidden_units']
        num_actor_outputs = ac_common_params['num_actor_outputs']
        final_layer_activation = ac_common_params['final_layer_activation']
        
        mean = np.array([NOISE_MIN, BETA_MIN, BSR_MIN], dtype=np.float32)[:input_dim]
        variance = np.power(np.array([NOISE_MAX - NOISE_MIN, 
                                              BETA_MAX - BETA_MIN, 
                                              BSR_MAX - BSR_MIN], dtype=np.float32) , 2)[:input_dim]

        inputs = layers.Input(shape = (input_dim), name = 'input_layer')
        input_normalization = layers.Normalization(axis = -1, mean = mean, variance = variance) (inputs)

        common = None
        for neurons in ac_common_hidden_units:
            if common is None:
                inp = input_normalization
            else:
                inp = common
            common = layers.Dense(neurons, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())(inp)
        if (common is None):
            common = inputs

        outputs = []
        
        actor_linear_hidden_units = ac_common_params['Actor']['linear_hidden_units']
        for idx_output in range(num_actor_outputs):
            actor_output = None
            for neurons in actor_linear_hidden_units:
                if actor_output is None:
                    inp = common
                else:
                    inp = actor_output
                actor_output = layers.Dense(neurons, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(inp)
            if (actor_output is None):
                actor_output = common
            actor_output = layers.Dense(
                                output_dim[idx_output], 
                                activation = final_layer_activation[idx_output],
                                kernel_initializer=tf.keras.initializers.GlorotNormal())(actor_output)
            outputs.append(actor_output)

        critic_output = None
        critic_linear_hidden_units = ac_common_params['Critic']['linear_hidden_units']
        for neurons in critic_linear_hidden_units:
            if critic_output is None:
                inp = common
            else:
                inp = critic_output
            critic_output = layers.Dense(neurons, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(inp)
        if (critic_output is None):
            critic_output = common
        critic_output = layers.Dense(
                            output_dim[-1],
                            activation = final_layer_activation[-1],
                            kernel_initializer=tf.keras.initializers.HeNormal())(critic_output)
        outputs.append(critic_output)
        return tf.keras.Model(inputs, outputs, name = name)

    def step(self):
        raise ValueError("Step needs to be implemented by the agent")

    def get_state_size(self):
        return self.environment.get_state_size()

    def get_action_size(self) -> np.ndarray:
        env_type = type(self.environment.action_space)
        if (env_type == gym.spaces.Discrete):
            return np.array([self.environment.action_space.n], dtype=np.int32)
        elif (env_type == gym.spaces.MultiDiscrete):
            return self.environment.action_space.nvec

    def track_episodes_data(self):
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes = None, save_and_print_results = True):
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
        
        if self.config.save_model: self.localy_save_policy()

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory = None, experience = None):
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.ad_experience(*experience)

    def soft_update_of_target_network(self, local_model, target_model, tau):
        local_model_weights = np.array(local_model.get_weights())
        target_model_weights = np.array(target_model.get_weights())
        target_model_weights = tau * local_model_weights + (1 - tau) * target_model_weights
        target_model.set_weights(target_model_weights)
    
    def setup_logger(self):
        filename = "Training.log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except: pass
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    @staticmethod
    def copy_model_over(from_model, to_model) -> None:
        """Copies model parameters from from_model to to_model"""
        to_model.set_weights(from_model.get_weights())
