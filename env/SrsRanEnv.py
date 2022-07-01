from multiprocessing import shared_memory
from tokenize import Number
import gym
from gym import spaces
import numpy as np

from env.DecoderEnv import BaseEnv

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

class SrsRanEnv(BaseEnv):
    def __init__(self,
                input_dims = 3,
                penalty = 15,
                policy_output_format = "mcs_prb_joint",
                title = "srsRAN Environment",
                verbose = 0) -> None:
        super(SrsRanEnv, self).__init__(
            input_dims = input_dims,
            penalty = penalty,
            policy_output_format = policy_output_format,
            title = title, 
            verbose = verbose)

    def setup(self, agent_idx, total_agents):
        super().setup(agent_idx, total_agents)
        self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
        observation_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
        self.observation_nd_array = observation_nd_array[agent_idx * 4: (agent_idx+1)*4]

        self.shm_action = shared_memory.SharedMemory(create=False, name='action')
        action_nd_array = np.ndarray(shape=(3 * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
        self.action_nd_array = action_nd_array[agent_idx * 3: (agent_idx + 1) * 3]

        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * 4: (agent_idx + 1) * 4]

        
    def step(self, action):        
        mcs, prb = super().translate_action(action)
        self.action_nd_array[:] = np.array([1, mcs, prb], dtype=np.int32)
        if (self.verbose == 1):
            print('{} - Act: {}'.format(str(self), action))
        if (prb > 0):
            while self.result_nd_array[0] == 0:
                pass
            result = self.result_nd_array[1:]
            self.result_nd_array[0] = 0
        else:
            result = np.array([True, 1, 0])
        crc, decoding_time, tbs = result
        reward = super().get_reward(mcs, prb, crc, decoding_time, tbs)
        print('{} - {}'.format(str(self), result.tolist() + [reward]))
        return super().get_agent_result(reward, mcs, prb, crc, decoding_time)
        

    def reset(self):
        while self.observation_nd_array[0] == 0:
            pass
        super().set_observation(self.observation_nd_array[1:][:self.input_dims])
        if (self.verbose == 1):
            print('{} - Obs: {}'.format(str(self), self.observation))
        self.observation_nd_array[0] = 0
        return super().get_observation()
        