import numpy as np

class Buffer:
    def __init__(self, buffer_capacity=64, batch_size=64, num_states = 2, num_actions = 1):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]

        self.buffer_counter += 1

    def sample(self, tf):
        state_batch = tf.convert_to_tensor(self.state_buffer)
        action_batch = tf.convert_to_tensor(self.action_buffer)
        reward_batch = tf.convert_to_tensor(self.reward_buffer)
        reward_batch = tf.cast(reward_batch, dtype = tf.float32)

        return state_batch, action_batch, reward_batch
