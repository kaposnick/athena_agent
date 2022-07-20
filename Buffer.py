import numpy as np

class EpisodeBuffer:
    def __init__(self, buffer_capacity=64, batch_size=64, num_states = 2, num_actions = 1):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))

    def reset_episode(self):
        self.episode_start_index = self.buffer_counter

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]

        self.buffer_counter += 1

    def sample(self, tf):
        valid_indices = np.arange(self.episode_start_index, self.episode_start_index + self.batch_size)
        valid_indices = valid_indices % self.buffer_capacity

        state_batch = tf.convert_to_tensor(self.state_buffer[valid_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[valid_indices])
        action_batch = tf.cast(action_batch, dtype = tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[valid_indices])
        reward_batch = tf.cast(reward_batch, dtype = tf.float32)

        return state_batch, action_batch, reward_batch

    def sample_sil(self, tf, critic):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        valid_indices = np.arange(0, record_range)

        values  = critic(self.state_buffer[valid_indices], training = False)
        rewards = self.reward_buffer[valid_indices]
        priorities = np.squeeze(np.maximum(rewards - values, 0))
        probabilities = priorities / np.sum(priorities)

        batch_indices = np.random.choice(valid_indices, self.batch_size, p = probabilities)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        action_batch = tf.cast(action_batch, dtype = tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype = tf.float32)

        return state_batch, action_batch, reward_batch

