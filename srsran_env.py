import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
from common_utils import MODE_SCHEDULING_ATHENA, MCS_SPACE, PRB_SPACE, PROHIBITED_COMBOS, I_MCS_TO_I_TBS, to_tbs


import time

class SrsRanEnv():
    def __init__(self,
                context_size = 2,
                action_size = 2,
                penalty = 1,
                title = "srsRAN Environment",
                verbose = 0,
                decode_deadline = 3000,
                scheduling_mode = MODE_SCHEDULING_ATHENA) -> None:
        self.context_size = context_size
        self.action_size = action_size
        self.penalty = penalty
        self.title = title
        self.verbose = verbose
        self.decode_deadline = decode_deadline
        self.scheduling_mode = scheduling_mode
        self.create_mcs_prb_array()

    def create_mcs_prb_array(self):
        self.random_action_idx = 0

        self.mapping_array = []
        for mcs in MCS_SPACE:
            for prb in PRB_SPACE:
                combo = ( I_MCS_TO_I_TBS[int(mcs)], int(prb) - 1)
                if combo in PROHIBITED_COMBOS:
                    continue
                self.mapping_array.append(
                    {   
                        'tbs': to_tbs(int(mcs), int(prb)),
                        'mcs': mcs,
                        'prb': prb
                    }
                )
        self.mapping_array = sorted(self.mapping_array, key = lambda el: (el['tbs'], el['mcs']))
        self.action_array = [np.array([x['mcs'], x['prb']]) for x in self.mapping_array] # sort by tbs/mcs
        self.action_array = np.array(self.action_array)        

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
        self.agent_idx = agent_idx
        self.set_title('worker_{}'.format(agent_idx))
        observation_size = 6        
        self.shm_observation = shared_memory.SharedMemory(create=False, name='observation')
        observation_nd_array = np.ndarray(shape=(observation_size * total_agents), dtype=np.int32, buffer=self.shm_observation.buf)
        self.observation_nd_array = observation_nd_array[agent_idx * observation_size: (agent_idx+1)*observation_size] # crc, tti, cpu, snr, bsr

        action_size = 3
        self.shm_action = shared_memory.SharedMemory(create=False, name='action')
        action_nd_array = np.ndarray(shape=(action_size * total_agents), dtype=np.int32, buffer = self.shm_action.buf)
        self.action_nd_array = action_nd_array[agent_idx * action_size: (agent_idx + 1) * action_size]

        verify_action_size = 2
        self.shm_verify_action = shared_memory.SharedMemory(create=False, name='verify_action')
        verify_action_nd_array = np.ndarray(shape=(verify_action_size * total_agents), dtype=np.int32, buffer = self.shm_verify_action.buf)
        self.verify_action_nd_array = verify_action_nd_array[agent_idx * verify_action_size: (agent_idx + 1) * verify_action_size]

        reward_size = 9
        self.shm_reward = shared_memory.SharedMemory(create=False, name='result')
        result_nd_array = np.ndarray(shape=(reward_size * total_agents), dtype=np.int32, buffer = self.shm_reward.buf)
        self.result_nd_array = result_nd_array[agent_idx * reward_size: (agent_idx + 1) * reward_size]

    def is_context_valid(self) -> bool:
        cpu, snr = self.observation
        is_valid = (cpu >= 0 and cpu <= 1000)
        is_valid = is_valid & (snr >= 0 and snr <=80)
        return is_valid

    def receive_context(self):
        with self.cond_observation:
            while self.observation_nd_array[0] == 0:
                self.cond_observation.wait(0.001)
        self.observation_nd_array[0] = 0 
        self.timestamp = self.current_timestamp()
        
        # observation_nd_array: crc, tti, cpu, snr, bsr, gain
        self.tti = self.observation_nd_array[1]        
        cpu, snr = self.observation_nd_array[2:4].astype(np.float32)
        self.bsr = self.observation_nd_array[4]
        self.gain = self.observation_nd_array[5]
        return np.array([cpu, snr / 1000], dtype=np.float32)

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
        mcs, prb = action
        self.apply_action(mcs, prb)
        verify_action = self.verify_action()            
        if (not verify_action):
            return None, None, True, None
        crc, decoding_time, tbs, mcs_res, prb_res, snr_res, noise_dbm, snr_custom = self.receive_reward()
        reward, _ = self.get_reward(mcs_res, prb_res, crc, decoding_time, tbs)
        cpu, snr = self.get_observation()
        result = self.get_agent_result(reward, mcs_res, prb_res, crc, decoding_time, tbs, snr, cpu, snr_res / 1000, noise_dbm / 1000, snr_custom / 1000)
        result[3]['modified'] = mcs_res != mcs or prb_res != prb            
        result[3]['tti'] = self.tti
        result[3]['hrq'] = self.agent_idx
        result[3]['timestamp'] = self.timestamp
        result[3]['gain'] = self.gain
        result[3]['bsr'] = self.bsr
        return result        

    def reset(self):        
        context = self.receive_context()
        self.set_observation(context)
        return self.get_observation()

    def current_timestamp(self):
        return round(time.time() * 1000)

    def get_reward(self, mcs, prb, crc, decoding_time, tbs = None):
        reward = 0
        tbs = None
        if ( prb > 0 and (I_MCS_TO_I_TBS[mcs], prb - 1) in PROHIBITED_COMBOS):
            reward = -1 * self.penalty
            return reward, None
        else:
            if tbs is None:
                tbs = to_tbs(mcs, prb)
            if (not crc or decoding_time > self.decode_deadline):
                reward = -1 * self.penalty
            else:
                reward = (tbs / ( 8 * 1024))
        return reward, tbs

    def get_agent_result(self, reward, mcs, prb, crc, decoding_time, tbs, snr, cpu, snr_res, noise_dbm, snr_custom):
        info = {'mcs': mcs, 'prb': prb, 
                'crc': crc, 'dec_time': decoding_time, 
                'tbs': tbs, 'snr': snr, 'reward': reward, 'cpu': cpu, 'snr_decode': snr_res, 'noise_decode': noise_dbm, 'snr_custom': snr_custom}
        return None, reward, True, info
    
    def set_title(self, title):
        self.title = title

    def get_title(self):
        return self.title
    
    def set_observation(self, observation):
        self.observation = observation
    
    def get_observation(self):
        return self.observation
        
    def __str__(self) -> str:
        return self.title