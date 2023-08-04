import numpy as np
from scipy.spatial import distance
from common_utils import to_tbs

class DDPGAgent():
    def __init__(self, tf, context_size, action_size) -> None:
        # action_size: refers to the size of the action space. 1 is for MCS only, 2 is for PRB and MCS.
        self.tf = tf
        self.context_size = context_size
        self.action_size = action_size
        self.actor = None
        self.critic = None

        self.mcs_prb_min = np.array([0, 1], dtype=np.float32)
        self.mcs_prb_max = np.array([24, 45], dtype=np.float32)
        self.mcs_min     = np.array([0], dtype=np.float32)
        self.mcs_max     = np.array([24], dtype=np.float32)

        self.context_min   = np.array([0, 18], dtype=np.float32)
        self.context_max   = np.array([1000, 49], dtype=np.float32)

        if (self.action_size) == 1:
            self.weights = np.array([1, 120])
        elif (self.action_size) == 2:
            self.weights = np.array([1, 1])

    def load_actor_weights(self, path):
        self.actor.load_weights(path)

    def load_critic_weights(self, path):
        self.critic.load_weights(path)

    def set_action_array(self, action_array):
        self.action_array = action_array
        self.l2_norms = np.zeros(shape=(len(self.action_array)))        

    def normalize_action(self, action):
        if (self.action_size == 1):
            return (action - self.mcs_min) / (self.mcs_max - self.mcs_min)
        elif (self.action_size == 2):
            return (action - self.mcs_prb_min) / (self.mcs_prb_max - self.mcs_prb_min)
        raise Exception("Unknown context")
    
    def denormalize_action(self, action):
        if (self.action_size == 1):
            return action * (self.mcs_max - self.mcs_min) + self.mcs_min
        elif (self.action_size == 2):
            return action * (self.mcs_prb_max - self.mcs_prb_min) + self.mcs_prb_min
        raise Exception("Unknown context")
    
    def normalize_context(self, context):
        return (context - self.context_min) / (self.context_max - self.context_min)

    def denormalize_context(self, context):
        return context * (self.context_max - self.context_min) + self.context_min
    
    def tidy_action(self, action):
        if (self.action_size == 1):
            return np.array([action[0].numpy(), 45])
        elif (self.action_size == 2):
            return action
        raise Exception("Unknown context")
    
    def adjust_action_for_critic(self, action):
        if (self.action_size == 1):
            return np.expand_dims(action[:,0], axis=1)
        elif (self.action_size == 2):
            return action
        raise Exception("Unknown context")            
    
    def __call__(self, context, k=9):
        tf = self.tf
        context = self.normalize_context(context)
        context = tf.convert_to_tensor([context], dtype=tf.float32)
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
    
    def readjust_to_demand(self, mcs, prb, bsr):
        mcs = int(mcs)
        prb = int(prb)
        tbs = to_tbs(mcs, prb)
        if (tbs <= bsr):
            return mcs, prb
        _candidate_mcs = 0
        while (_candidate_mcs <= mcs):
            tbs = to_tbs(_candidate_mcs, prb)
            if (tbs >= bsr):
                break
            _candidate_mcs += 1
        return _candidate_mcs, prb

    def load_actor(self):
        keras = self.tf.keras
        layers = keras.layers

        context_input = keras.Input(shape = (self.context_size))
        x = layers.Dense(16, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (context_input)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)    
        norm_params = layers.Dense(self.action_size, activation='sigmoid', kernel_initializer = keras.initializers.HeNormal())(x)
        self.actor = keras.Model(context_input, norm_params)
        return self.actor
    
    def load_critic(self):
        keras = self.tf.keras
        layers = keras.layers
        context_input = keras.Input(shape = (self.context_size))
        action_input = keras.Input(shape = (self.action_size))
        x = layers.Concatenate()([context_input, action_input])
        x = layers.Dense(16, activation = 'relu', kernel_initializer     = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        q = layers.Dense(1, kernel_initializer = keras.initializers.HeNormal()) (x)
        self.critic = keras.Model(inputs = [context_input, action_input], outputs=q)
        return self.critic


    



