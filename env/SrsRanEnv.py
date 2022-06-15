import gym
from gym import spaces
import numpy as np

PROHIBITED_COMBOS = [(0, 1), (0, 2), (0,3), 
                  (1, 0), (1, 1),
                  (2, 0), (2, 1),
                  (3, 0), (4, 0), (5, 0), 
                  (6, 1)]

PRB_SPACE = np.array(
                    [0, 1, 2, 3, 4, 5, 6, 8, 9, 
                      10, 12, 15, 16, 18, 
                      20, 24, 25, 27, 
                      30, 32, 36, 40, 45], dtype = np.float16)
MCS_SPACE = np.arange(0, 25,  dtype=np.float16)

class SrsRanEnv():
    def __init__(self,
                 title,
                 penalty = 15) -> None:
        super(SrsRanEnv, self).__init__()
        self.penalty = penalty
        self.title = title

        # Define a 1-D observation space
        self.observation_shape = (3,)
        self.observation_space = spaces.Box(
                            low  = np.zeros(self.observation_shape),
                            high = np.zeros(self.observation_shape), 
                            dtype = np.float32)       
        
        n_actions = (len(MCS_SPACE), len(PRB_SPACE))        
        self.action_space = spaces.MultiDiscrete(n_actions)

    def set_queues(self, input_observation_queue, 
                         output_action_queue, 
                         input_reward_queue):
        self.observation_queue = input_observation_queue
        self.action_queue = output_action_queue
        self.reward_queue = input_reward_queue

    def get_environment_title(self):
        return self.title

    def __str__(self) -> str:
        return self.title
    
    def translate_action(self, action):
        return int(MCS_SPACE[action[0]]), int(PRB_SPACE[action[1]])
    
    def run(self):
        while(True):
            self.reset()
            # print('{} - Obs: {}'.format(str(self), self.observation))
            action = (10 + self.n, 1 + self.n)
            _, reward, _, _ = self.step(action)

    def reward(self, crc, decoding_time, tbs):
        reward = 0
        if (crc == True and decoding_time <= 3000):
            reward = tbs / (8 * 1024) # in KBs
        else:
            reward = -1 * self.penalty
        return reward
        
    def step(self, action):        
        # mcs, prb = self.translate_action(action)
        mcs, prb = action[0], action[1]
        self.action_queue.put([mcs, prb])
        reward = self.reward_queue.get(block = True)
        print('{} - Rew: {}'.format(str(self), reward))
        
        tti, crc, decoding_time, tbs = reward
        result = self.reward(crc, decoding_time, tbs)
        return None, result, True, {} 

    def reset(self):
        self.observation = self.observation_queue.get(block = True) # noise, beta, bsr
        return self.observation
        