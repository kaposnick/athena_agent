import random
import numpy as np
from numpy.random import choice
from collections import namedtuple, deque

class EpisodeBuffer:
    def __init__(self, buffer_size=64, batch_size=64, experiences_per_sampling = 2, compute_weights = False, state_size = 2, num_actions = 1):
        self.state_size = state_size
        self.num_actions = num_actions

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences_per_sampling = experiences_per_sampling
        self.compute_weights = compute_weights
        self.alpha = 0.5
        self.alpha_decay_rate = .5 
        self.beta = .5
        self.beta_growth_rate = 1.001
        self.experience_count = 0

        self.experience = namedtuple("Experience",
            field_names = ["state", "action", "reward"])
        self.data = namedtuple("Data",
            field_names = ["priority", "probability", "weight", "index"])
        
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
            N = min(self.experience_count, self.buffer_size)
            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                updated_weight = ((N * updated_priority) ** (-self.beta)) / self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1
            
            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority ** self.alpha - old_priority ** self.alpha
            updated_probability = td[0] ** self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index)
            self.memory_data[index] = data

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
        N = min(self.experience_count, self.buffer_size)
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
        print("sum_prob before: ", sum_prob_before)
        print("sum_prob after:  ", sum_prob_after)

    def record(self, state, action, reward):
        index = self.experience_count % self.buffer_size
        self.experience_count += 1

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_aplha -= temp.priority ** self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key = operator.itemgetter(1)).priority
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[index].weight = 0
                    self.weights_max = max(self.memory_data.items(), key = operator.itemgetter(2)).weight

        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward)
        self.memory[index] = e
        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d

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

