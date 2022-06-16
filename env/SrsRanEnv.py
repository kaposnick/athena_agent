from multiprocessing import shared_memory
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
                 verbose = 0,
                 penalty = 15) -> None:
        super(SrsRanEnv, self).__init__()
        self.penalty = penalty
        self.title = title
        self.verbose = verbose

        # Define a 1-D observation space
        self.observation_shape = (3,)
        self.observation_space = spaces.Box(
                            low  = np.zeros(self.observation_shape),
                            high = np.zeros(self.observation_shape), 
                            dtype = np.float32)       
        
        n_actions = (len(MCS_SPACE), len(PRB_SPACE))        
        self.action_space = spaces.MultiDiscrete(n_actions)

    def setup(self, agent_idx, total_agents):
        self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
        observation_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
        self.observation_nd_array = observation_nd_array[agent_idx * 4: (agent_idx+1)*4]

        self.shm_action = shared_memory.SharedMemory(create=False, name='action')
        action_nd_array = np.ndarray(shape=(3 * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
        self.action_nd_array = action_nd_array[agent_idx * 3: (agent_idx + 1) * 3]

        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * 4: (agent_idx + 1) * 4]

        self.title = 'worker_{}'.format(agent_idx)

    def get_environment_title(self):
        return self.title

    def __str__(self) -> str:
        return self.get_environment_title()
    
    def translate_action(self, action):
        return int(MCS_SPACE[action[0]]), int(PRB_SPACE[action[1]])

    def reward(self, crc, decoding_time, tbs):
        reward = 0
        if (crc == True and decoding_time <= 3000):
            reward = tbs / (8 * 1024) # in KBs
        else:
            reward = -1 * self.penalty
        return reward
        
    def step(self, action):        
        mcs, prb = self.translate_action(action)
        self.action_nd_array[:] = np.array([1, mcs, prb], dtype=np.int32)
        if (self.verbose == 1):
            print('{} - Act: {}'.format(str(self), action))
        while self.result_nd_array[0] == 0:
            pass
        result = self.result_nd_array[1:]
        self.result_nd_array[0] = 0

        if (self.verbose == 1):
            print('{} - Res: {}'.format(str(self), result))
        
        crc, decoding_time, tbs = result
        result = self.reward(crc, decoding_time, tbs)
        return None, result, True, {} 

    def reset(self):
        while self.observation_nd_array[0] == 0:
            pass
        self.observation = self.observation_nd_array[1:]
        if (self.verbose == 1):
            print('{} - Obs: {}'.format(str(self), self.observation))
        self.observation_nd_array[0] = 0
        return self.observation
        