import numpy as np
import random

class SumTree:
    def __init__(self, size) -> None:
        self.nodes = np.zeros(2 * size - 1)
        self.data =  np.zeros(size, dtype = object)

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def add(self, priority, data):
        self.data[self.count] = data
        self.update(self.count, priority)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def update(self, tree_idx, priority):
        idx = tree_idx + self.size - 1
        change = priority - self.nodes[idx]
        self.nodes[idx] = priority
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def get(self, cumsum):
        assert cumsum <= self.total
        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]
        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]

class ProiritizedReplayBuffer:
    def __init__(self, state_size, num_actions, buffer_size, batch_size, eps=1e-2, alpha=.1, beta=.1) -> None:
        self.tree = SumTree(size = buffer_size)
        self.eps = eps
        self.alpha = alpha # determines how much prioritization is used, a = 0 corresponding to the uniform case
        self.beta = beta   # determines the amount of importance sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps # priorirty for new samples, init as eps
        self.batch_size = batch_size

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

        self.state = np.zeros(shape = (buffer_size, state_size))
        self.action = np.zeros(shape = (buffer_size, num_actions))
        self.reward = np.zeros(shape = (buffer_size, 1))

    def add(self, state, action, reward):
        self.tree.add(self.max_priority, self.count)
        self.state[self.count] = state
        self.action[self.count] = action
        self.reward[self.count] = reward
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, tf):
        sample_idxs, tree_idxs = [], []
        priorities = np.zeros(shape=(self.batch_size, 1), dtype = np.float32)

        # To sample a minimbatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally, the transitions that correspond
        # to each of these sampled values are retrieved from the tree
        segment = self.tree.total / self.batch_size
        for i in range(self.batch_size):
            a, b = segment * i, segment * (i+1)
            cumsum = random.uniform(a, b)

            # sample_idx is a sample index in buffer, needed furthere to sample actual transitions
            # tree_idx is an index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / np.max(weights)

        state_batch  = tf.reshape(tf.convert_to_tensor(self.state[sample_idxs], dtype = tf.float32), [-1, self.state_size])
        action_batch = tf.reshape(tf.convert_to_tensor(self.action[sample_idxs], dtype = tf.float32), [-1, self.num_actions])
        reward_batch = tf.reshape(tf.convert_to_tensor(self.reward[sample_idxs], dtype = tf.float32), [-1, 1])

        return state_batch, action_batch, reward_batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
