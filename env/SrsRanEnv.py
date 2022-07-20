import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

from env.DecoderEnv import BaseEnv

class SrsRanEnv(BaseEnv):
    def __init__(self,
                input_dims = 3,
                penalty = 15,
                policy_output_format = "mcs_prb_joint",
                title = "srsRAN Environment",
                verbose = 0,
                in_scheduling_mode = True) -> None:
        super(SrsRanEnv, self).__init__(
            input_dims = input_dims,
            penalty = penalty,
            policy_output_format = policy_output_format,
            title = title, 
            verbose = verbose, 
            in_scheduling_mode = in_scheduling_mode)
             

    def presetup(self, inputs):
        if (self.in_scheduling_mode):
            cond_observation = inputs['cond_observation']
            self.cond_observation = cond_observation
            
            cond_action = inputs['cond_action']
            self.cond_action = cond_action
        
        cond_reward = inputs['cond_reward']
        self.cond_reward = cond_reward

    def setup(self, agent_idx, total_agents):
        super().setup(agent_idx, total_agents)
        if (self.in_scheduling_mode):
            self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
            observation_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
            self.observation_nd_array = observation_nd_array[agent_idx * 4: (agent_idx+1)*4]

            self.shm_action = shared_memory.SharedMemory(create=False, name='action')
            action_nd_array = np.ndarray(shape=(3 * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
            self.action_nd_array = action_nd_array[agent_idx * 3: (agent_idx + 1) * 3]

        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(6 * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * 6: (agent_idx + 1) * 6]

        
    def step(self, action):
        if (self.in_scheduling_mode):
            mcs, prb = super().translate_action(action)
            with self.cond_action:
                self.action_nd_array[:] = np.array([1, mcs, prb], dtype=np.int32)
                self.cond_action.notify()
            if (self.verbose == 1):
                print('{} - Act: {}'.format(str(self), action))
            if (prb > 0):
                with self.cond_reward:
                    while self.result_nd_array[0] == 0:
                        self.cond_reward.wait(0.1)
                    pass
                result = self.result_nd_array[1:]
                self.result_nd_array[0] = 0
            else:
                result = np.array([True, 1, 0, 0, 0])
            crc, decoding_time, tbs, mcs_res, prb_res = result
            if (mcs_res != mcs or prb_res != prb):
                string_inside = 'Wrong combination of {}, {}'.format( (mcs, prb), (mcs_res, prb_res))
                print('{} - {}'.format(str(self), string_inside))
            reward, _ = super().get_reward(mcs, prb, crc, decoding_time, tbs)
            if (self.verbose == 1):
                print('{} - {}'.format(str(self), result.tolist() + [reward]))
            
            result = super().get_agent_result(reward, mcs, prb, crc, decoding_time, tbs)
        else:
            with self.cond_reward:
                while self.result_nd_array[0] == 0:
                        self.cond_reward.wait(0.1)
            result = self.result_nd_array[1:]
            self.result_nd_array[0] = 0
            crc, decoding_time, tbs, mcs, prb = result
            result = super().get_agent_result(None, mcs, prb, crc, decoding_time, tbs)
        return result
        

    def reset(self):
        with self.cond_observation:
            while self.observation_nd_array[0] == 0:
                self.cond_observation.wait(0.1)
        self.observation_nd_array[0] = 0
        super().set_observation(self.observation_nd_array[1:][:self.input_dims].tolist())
        if (self.verbose == 1):
            print('{} - Obs: {}'.format(str(self), self.observation))
        return super().get_observation()
        