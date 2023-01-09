import random
import numpy as np
from numpy.random import choice
from collections import namedtuple, deque
from .SumTree import SumTree

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_size=2, num_actions=1) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.num_actions = num_actions
        self.state = np.zeros(shape = (buffer_size, state_size))
        self.action = np.zeros(shape = (buffer_size, num_actions))
        self.reward = np.zeros(shape = (buffer_size, 1))
        self.count = 0
        self.real_size = 0

    def add(self, state, action, reward):
        self.state[self.count] = state
        self.action[self.count] = action
        self.reward[self.count] = reward
        self.count = (self.count + 1) % self.buffer_size
        self.real_size = min(self.buffer_size, self.real_size + 1)

    def sample(self, tf):
        idxs = np.random.choice(self.real_size, self.batch_size, replace=False)
        state, action, reward = self.convert_to_tensors(tf, idxs)
        return state, action, reward

    def convert_to_tensors(self, tf, idxs):
        state  = tf.reshape(tf.convert_to_tensor(self.state[idxs], dtype = tf.float32), [-1, self.state_size])
        action = tf.reshape(tf.convert_to_tensor(self.action[idxs], dtype = tf.float32), [-1, self.num_actions])
        reward = tf.reshape(tf.convert_to_tensor(self.reward[idxs], dtype = tf.float32), [-1, 1])
        return state, action, reward

class PERBuffer_Proportional(ReplayBuffer):
    def __init__(self, state_size, num_actions, buffer_size, batch_size, eps=1e-2, alpha=.1, beta=.1, alpha_decay_rate=1, beta_growth_rate=1) -> None:
        super().__init__(buffer_size=buffer_size, batch_size=batch_size, state_size=state_size, num_actions=num_actions)
        self.tree = SumTree(size = buffer_size)
        self.eps = eps
        self.alpha = alpha # determines how much prioritization is used, a = 0 corresponding to the uniform case
        self.beta = beta   # determines the amount of importance sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps # priorirty for new samples, init as eps
        self.alpha_decay_rate = alpha_decay_rate
        self.beta_growth_rate = beta_growth_rate

    def add(self, state, action, reward):
        self.tree.add(self.max_priority, self.count)
        super().add(state, action, reward)

    def sample(self, tf):
        idxs, tree_idxs = [], []
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
            idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / np.max(weights)

        state, action, reward = super().convert_to_tensors(tf, idxs)
        return state, action, reward, weights, tree_idxs

    def update_priorities(self, priorities, data_idxs):
        priorities = np.squeeze(priorities)
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if (self.beta > 1):
            self.beta = 1

    def update_memory_sampling(self):
        # n/a/
        pass

class PERBuffer_RankBased(ReplayBuffer):
    def __init__(self, buffer_size=64, batch_size=64, experiences_per_sampling = 2, compute_weights = False, state_size = 2, num_actions = 1):
        super.__init__(buffer_size=buffer_size, batch_size=batch_size, state_size=state_size, num_actions=num_actions)
        self.experiences_per_sampling = experiences_per_sampling
        self.compute_weights = compute_weights
        self.alpha = 0.5
        self.alpha_decay_rate = .5 
        self.beta = .5
        self.beta_growth_rate = 1.001

        self.data_priority = np.zeros(shape = (self.buffer_size, 1))
        self.data_probability = np.zeros(shape = (self.buffer_size, 1))
        self.data_weight = np.zeros(shape = (self.buffer_size, 1))
        self.data_idx = np.reshape(np.arange(self.buffer_size), newshape = (-1, 1))
        
        indices = []
        datas   = []
        for i in range(buffer_size):
            indices.append(i)
            d = self.data(0,0,0,i)
            datas.append(d)
        self.memory = {key: self.experience for key in indices}
        self.memory_data = {key: data for key, data in zip(indices, datas)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = self.real_size
            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                updated_weight = ((N * updated_priority) ** (-self.beta)) / self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1
            
            old_priority = self.data_priority[index]
            self.priorities_sum_alpha += updated_priority ** self.alpha - old_priority ** self.alpha
            updated_probability = td[0] ** self.alpha / self.priorities_sum_alpha
            self.data_priority[index] = updated_priority
            self.data_probability[index] = updated_probability
            self.data_weight[index] = updated_weight
            self.data_idx[index] = index

    def update_memory_sampling(self):
        """Randomly sample X batches of experiences from memory."""
        # X is the number of steps before updating memory
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(
            self.memory_data,
            [data.probability for data in values], 
            k = self.experiences_per_sampling
        )
        self.sampled_batches = [
            random_values[i:i+self.batch_size] 
                for i in range(0, len(random_values), self.batch_size)
        ]
    
    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if (self.beta > 1):
            self.beta = 1
        N = min(self.count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority ** self.alpha
        
        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority ** self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            if self.compute_weights:
                weight = ((N * element.probability) ** (-self.beta)) / self.weights_max
            d = self.data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d

    def add(self, state, action, reward):
        if (self.count + 1) > self.buffer_size:
            temp = self.memory_data[self.count]
            self.priorities_sum_aplha -= temp.priority ** self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[self.count].priority = 0
                self.priorities_max = max(self.memory_data.items(), key = operator.itemgetter(1)).priority
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[self.count].weight = 0
                    self.weights_max = max(self.memory_data.items(), key = operator.itemgetter(2)).weight

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        self.save_data(priority, probability, weight, self.count)
        super().add(state, action, reward)

    def save_data(self, priority, probability, weight, index):
        self.data_priority[self.count] = priority
        self.data_probability[self.count] = probability
        self.data_weight[self.count] = weight
        self.data_idx[self.count] = index

    def sample(self, tf):
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        states = []
        actions = []
        rewards = []
        weights = []
        indices = []
        for data in sampled_batch:
            experience = self.memory.get(data.index)
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            weights.append(data.weight)
            indices.append(data.index)

        state_batch  = tf.reshape(tf.convert_to_tensor(states, dtype = tf.float32), [-1, self.state_size])
        action_batch = tf.reshape(tf.convert_to_tensor(actions, dtype = tf.float32), [-1, self.num_actions])
        reward_batch = tf.reshape(tf.convert_to_tensor(rewards, dtype = tf.float32), [-1, 1])

        return state_batch, action_batch, reward_batch, weights, indices

