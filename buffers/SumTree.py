import numpy as np

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