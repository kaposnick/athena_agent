import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from common_utils import MODE_SCHEDULING_AC, MODE_SCHEDULING_RANDOM, denormalize_state

from env.DecoderEnv import BaseEnv

class SrsRanEnv(BaseEnv):
    def __init__(self,
                input_dims = 3,
                penalty = 15,
                policy_output_format = "mcs_prb_joint",
                title = "srsRAN Environment",
                verbose = 0,
                scheduling_mode = MODE_SCHEDULING_AC) -> None:
        super(SrsRanEnv, self).__init__(
            input_dims = input_dims,
            penalty = penalty,
            policy_output_format = policy_output_format,
            title = title, 
            verbose = verbose, 
            scheduling_mode = scheduling_mode)
             

    def presetup(self, inputs):
        if (self.scheduling_mode):
            cond_observation = inputs['cond_observation']
            self.cond_observation = cond_observation
            
            cond_action = inputs['cond_action']
            self.cond_action = cond_action

            cond_verify_action = inputs['cond_verify_action']
            self.cond_verify_action = cond_verify_action
        
        cond_reward = inputs['cond_reward']
        self.cond_reward = cond_reward

    def setup(self, agent_idx, total_agents):
        super().setup(agent_idx, total_agents)
        if (self.scheduling_mode):
            self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
            observation_nd_array = np.ndarray(shape=(4 * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
            self.observation_nd_array = observation_nd_array[agent_idx * 4: (agent_idx+1)*4]

            self.shm_action = shared_memory.SharedMemory(create=False, name='action')
            action_nd_array = np.ndarray(shape=(3 * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
            self.action_nd_array = action_nd_array[agent_idx * 3: (agent_idx + 1) * 3]

            self.shm_verify_action = shared_memory.SharedMemory(create=False, name='verify_action')
            verify_action_nd_array = np.ndarray(shape=(2 * total_agents), dtype=np.int32, buffer = self.shm_verify_action.buf)
            self.verify_action_nd_array = verify_action_nd_array[agent_idx * 2: (agent_idx + 1) * 2]

        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(7 * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * 7: (agent_idx + 1) * 7]

        
    def step(self, action):
        if (self.scheduling_mode == MODE_SCHEDULING_AC or self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            mcs, prb = super().translate_action(action)
            with self.cond_action:
                self.action_nd_array[:] = np.array([1, mcs, prb], dtype=np.int32)
                self.cond_action.notify()
            if (self.verbose == 1):
                print('{} - Act: {}'.format(str(self), action))
            with self.cond_verify_action:
                while self.verify_action_nd_array[0] == 0:
                    self.cond_verify_action.wait(0.001)

            verify_action = self.verify_action_nd_array[1:]
            self.verify_action_nd_array[0] = 0
            if (not verify_action):
                return None, None, True, None
            with self.cond_reward:
                while self.result_nd_array[0] == 0:
                    self.cond_reward.wait(0.001)
                pass
            result = self.result_nd_array[1:]
            self.result_nd_array[0] = 0
            # if (prb > 0):
            # else:
            #     result = np.array([True, 1, 0, 0, 0, 0])
            crc, decoding_time, tbs, mcs_res, prb_res, snr = result
            snr_real = snr / 1000.0
            if (mcs_res != mcs or prb_res != prb and self.verbose == 1):
                string_inside = 'Wrong combination of {}, {}'.format( (mcs, prb), (mcs_res, prb_res))
                print('{} - {}'.format(str(self), string_inside))
            reward, _ = super().get_reward(mcs_res, prb_res, crc, decoding_time, tbs)
            if (self.verbose == 1):
                print('{} - {}'.format(str(self), result.tolist() + [reward]))
            cpu, snr_real = denormalize_state(super().get_observation())
            result = super().get_agent_result(reward, mcs_res, prb_res, crc, decoding_time, tbs, snr_real, cpu)
        else:
            with self.cond_reward:
                while self.result_nd_array[0] == 0:
                        self.cond_reward.wait(0.001)
            result = self.result_nd_array[1:]
            self.result_nd_array[0] = 0
            crc, decoding_time, tbs, mcs, prb, snr = result
            snr_real = snr / 1000.0
            cpu, snr_real = denormalize_state(super().get_observation())
            result = super().get_agent_result('', mcs, prb, crc, decoding_time, tbs, snr_real, cpu)
        return result
        

    def reset(self):
        with self.cond_observation:
            while self.observation_nd_array[0] == 0:
                self.cond_observation.wait(0.001)
        self.observation_nd_array[0] = 0
        state = self.observation_nd_array[1:][:self.input_dims].astype(np.float32)
        super().set_observation(state)
        if (self.verbose == 1):
            print('{} - Obs: {}'.format(str(self), self.observation))
        return super().get_observation()
        