import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
from common_utils import MODE_SCHEDULING_AC, MODE_SCHEDULING_RANDOM, denormalize_state

from env.DecoderEnv import BaseEnv
import time

class SrsRanEnv(BaseEnv):
    def __init__(self,
                input_dims = 3,
                penalty = 15,
                policy_output_format = "mcs_prb_joint",
                title = "srsRAN Environment",
                verbose = 0,
                decode_deadline = 3000,
                scheduling_mode = MODE_SCHEDULING_AC) -> None:
        super(SrsRanEnv, self).__init__(
            input_dims = input_dims,
            penalty = penalty,
            policy_output_format = policy_output_format,
            title = title, 
            verbose = verbose, 
            decode_deadline = decode_deadline,
            scheduling_mode = scheduling_mode)
             

    def presetup(self, inputs):
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
        self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
        observation_nd_array = np.ndarray(shape=(6 * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
        self.observation_nd_array = observation_nd_array[agent_idx * 6: (agent_idx+1)*6] # crc, tti, cpu, snr, bsr

        self.shm_action = shared_memory.SharedMemory(create=False, name='action')
        action_nd_array = np.ndarray(shape=(3 * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
        self.action_nd_array = action_nd_array[agent_idx * 3: (agent_idx + 1) * 3]

        self.shm_verify_action = shared_memory.SharedMemory(create=False, name='verify_action')
        verify_action_nd_array = np.ndarray(shape=(2 * total_agents), dtype=np.int32, buffer = self.shm_verify_action.buf)
        self.verify_action_nd_array = verify_action_nd_array[agent_idx * 2: (agent_idx + 1) * 2]

        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(9 * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * 9: (agent_idx + 1) * 9]

    def receive_state(self):
        with self.cond_observation:
            while self.observation_nd_array[0] == 0:
                self.cond_observation.wait(0.001)
        self.observation_nd_array[0] = 0 
        self.timestamp = self.current_timestamp()
        # observation_nd_array: crc, tti, cpu, snr, bsr, gain
        self.tti = self.observation_nd_array[1]        
        cpu, snr = self.observation_nd_array[2:4].astype(np.float32)
        self.gain = self.observation_nd_array[5]
        return np.array([cpu, snr / 1000], dtype=np.float32)
        # return self.observation_nd_array[2:4].astype(np.float32) # tti, cpu, snr

    def apply_action(self, mcs, prb):
        with self.cond_action:
            self.action_nd_array[:] = np.array([1, mcs, prb], dtype=np.int32)
            self.cond_action.notify()

    def verify_action(self):
        with self.cond_verify_action:
            while self.verify_action_nd_array[0] == 0:
                self.cond_verify_action.wait(0.001)
        verify_action = self.verify_action_nd_array[1:]
        self.verify_action_nd_array[0] = 0
        return verify_action

    def receive_reward(self):
        with self.cond_reward:
            while self.result_nd_array[0] == 0:
                self.cond_reward.wait(0.001)

        result = self.result_nd_array[1:]
        self.result_nd_array[0] = 0
        return result
        
    def step(self, action):
        if (self.scheduling_mode == MODE_SCHEDULING_AC or self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            if len(action) == 2:
                mcs, prb = action
            elif len(action) == 1:
                mcs, prb = super().translate_action(action[0])
            self.apply_action(mcs, prb)
            verify_action = self.verify_action()            
            if (not verify_action):
                return None, None, True, None
            crc, decoding_time, tbs, mcs_res, prb_res, snr_res, noise_dbm, snr_custom = self.receive_reward()
            reward, _ = super().get_reward(mcs_res, prb_res, crc, decoding_time, tbs)
            cpu, snr = super().get_observation()
            result = super().get_agent_result(reward, mcs_res, prb_res, crc, decoding_time, tbs, snr, cpu, snr_res / 1000, noise_dbm / 1000, snr_custom / 1000)
            result[3]['modified'] = mcs_res != mcs or prb_res != prb            
            result[3]['tti'] = self.tti
            result[3]['hrq'] = self.agent_idx
            result[3]['timestamp'] = self.timestamp
            result[3]['gain'] = self.gain
        else:
            mcs, prb = action
            self.apply_action(mcs, prb)
            verify_action = self.verify_action()            
            if (not verify_action):
                return None, None, True, None            
            crc, decoding_time, tbs, mcs_res, prb_res, snr_res, noise_dbm, snr_custom = self.receive_reward()
            cpu, snr = super().get_observation()
            result = super().get_agent_result('', mcs_res, prb_res, crc, decoding_time, tbs, snr, cpu, snr_res / 1000, noise_dbm / 1000, snr_custom / 1000)
            result[3]['modified'] = mcs_res != mcs or prb_res != prb            
            result[3]['tti'] = self.tti
            result[3]['hrq'] = self.agent_idx
            result[3]['timestamp'] = self.timestamp
            result[3]['gain'] = self.gain
        return result
        

    def reset(self):        
        state = self.receive_state()
        super().set_observation(state)
        return super().get_observation()

    def current_timestamp(self):
        return round(time.time() * 1000)
        