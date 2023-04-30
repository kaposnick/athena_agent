import numpy as np
from scipy.spatial import distance

class DDPGAgent():
    def __init__(self, tf, num_input, num_output) -> None:
        # num_output: refers to the size of the action space. 1 is for MCS only, 2 is for PRB and MCS.
        self.tf = tf
        self.num_input = num_input
        self.num_output = num_output
        self.actor = None
        self.critic = None

        self.mcs_prb_min = np.array([0, 1], dtype=np.float32)
        self.mcs_prb_max = np.array([24, 45], dtype=np.float32)
        self.mcs_min     = np.array([0], dtype=np.float32)
        self.mcs_max     = np.array([24], dtype=np.float32)

        self.state_min   = np.array([0, 18], dtype=np.float32)
        self.state_max   = np.array([1000, 49], dtype=np.float32)

        if (self.num_output) == 1:
            self.weights = np.array([1, 120])
        elif (self.num_output) == 2:
            self.weights = np.array([1, 1])

    def load_actor_weights(self, path):
        self.actor.load_weights(path)

    def load_critic_weights(self, path):
        self.critic.load_weights(path)

    def set_action_array(self, action_array):
        self.action_array = action_array
        self.l2_norms = np.zeros(shape=(len(self.action_array)))        

    def normalize_action(self, action):
        if (self.num_output == 1):
            return (action - self.mcs_min) / (self.mcs_max - self.mcs_min)
        elif (self.num_output == 2):
            return (action - self.mcs_prb_min) / (self.mcs_prb_max - self.mcs_prb_min)
        raise Exception("Unknown state")
    
    def denormalize_action(self, action):
        if (self.num_output == 1):
            return action * (self.mcs_max - self.mcs_min) + self.mcs_min
        elif (self.num_output == 2):
            return action * (self.mcs_prb_max - self.mcs_prb_min) + self.mcs_prb_min
        raise Exception("Unknown state")
    
    def normalize_state(self, state):
        return (state - self.state_min) / (self.state_max - self.state_min)

    def denormalize_state(self, state):
        return state * (self.state_max - self.state_min) + self.state_min
    
    def tidy_action(self, action):
        if (self.num_output == 1):
            return np.array([action[0].numpy(), 45])
        elif (self.num_output == 2):
            return action
        raise Exception("Unknown state")
    
    def adjust_action_for_critic(self, action):
        if (self.num_output == 1):
            return np.expand_dims(action[:,0], axis=1)
        elif (self.num_output == 2):
            return action
        raise Exception("Unknown state")            
    
    def __call__(self, state, k=9):
        tf = self.tf
        state = self.normalize_state(state)
        context = tf.convert_to_tensor([state], dtype=tf.float32)
        action_normalized = self.actor(context)[0]
        action = self.denormalize_action(action_normalized)
        action = self.tidy_action(action)
        for i, mcs_prb_comb in enumerate(self.action_array):
            self.l2_norms[i] = distance.euclidean(action, mcs_prb_comb, self.weights)
        partition = np.argpartition(self.l2_norms, k)
        k_closest_actions  = self.action_array[partition[:k]]
        k_closest_actions_normalized = self.normalize_action(k_closest_actions)
        k_closest_actions_normalized = self.adjust_action_for_critic(k_closest_actions_normalized)
        context_extended   = np.broadcast_to(context, (k, context.shape[1]))
        q_values           = self.critic([context_extended, k_closest_actions_normalized])
        argmin_q_value     = np.argmax(q_values)

        closest_action     = k_closest_actions[argmin_q_value]
        return str(action_normalized), int(closest_action[0]), int(closest_action[1])

    def load_actor(self):
        keras = self.tf.keras
        layers = keras.layers

        state_input = keras.Input(shape = (self.num_input))
        x = layers.Dense(16, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (state_input)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)    
        norm_params = layers.Dense(self.num_output, activation='sigmoid', kernel_initializer = keras.initializers.HeNormal())(x)
        self.actor = keras.Model(state_input, norm_params)
        return self.actor
    
    def load_critic(self):
        keras = self.tf.keras
        layers = keras.layers
        state_input = keras.Input(shape = (self.num_input))
        action_input = keras.Input(shape = (self.num_output))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(16, activation = 'relu', kernel_initializer     = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        q = layers.Dense(1, kernel_initializer = keras.initializers.HeNormal()) (x)
        self.critic = keras.Model(inputs = [state_input, action_input], outputs=q)
        return self.critic


    



