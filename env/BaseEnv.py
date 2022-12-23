from tokenize import Number
import numpy as np
import gym
from gym import spaces

from common_utils import MODE_SCHEDULING_AC

PROHIBITED_COMBOS = [(0, 0), (0, 1), (0,2), (0, 3), 
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1),
                  (3, 0), 
                  (4, 0), 
                  (5, 0), 
                  (6, 0)]

PRB_SPACE = np.array(
                    [1, 2, 3, 4, 5, 6, 8, 9, 
                      10, 12, 15, 16, 18, 
                      20, 24, 25, 27, 
                      30, 32, 36, 40, 45], dtype = np.float16)
# PRB_SPACE = np.array([45], dtype=np.float16)
MCS_SPACE =      np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ,24],  dtype=np.float16)  
I_MCS_TO_I_TBS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26])


class BaseEnv(gym.Env):
    def __init__(self, 
                input_dims: Number, 
                penalty: Number,
                policy_output_format: str,
                title: str, 
                verbose: Number,
                decode_deadline = 3000, 
                scheduling_mode = MODE_SCHEDULING_AC, 
                tbs_table_path = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json') -> None:
        super(BaseEnv, self).__init__()
        self.min_penalty = penalty
        self.policy_output_format = policy_output_format
        self.title = title
        self.verbose = verbose
        self.decode_deadline = decode_deadline
        self.scheduling_mode = scheduling_mode
        self.input_dims = input_dims
        self.observation_shape = (input_dims, )
        self.observation_space = spaces.Box(
            low = np.zeros(self.observation_shape),
            high = np.zeros(self.observation_shape),
            dtype = np.float32
        )

        import json
        with open(tbs_table_path) as tbs_json:
            self.tbs_table = json.load(tbs_json)  

        if (self.policy_output_format == "mcs_prb_joint"):
            self.mapping_array = []
            for mcs in MCS_SPACE:
                for prb in PRB_SPACE:
                    combo = ( I_MCS_TO_I_TBS[int(mcs)], int(prb) - 1)
                    if combo in PROHIBITED_COMBOS:
                        continue
                    self.mapping_array.append(
                        {   
                            'tbs': self.to_tbs(int(mcs), int(prb)),
                            'mcs': mcs,
                            'prb': prb
                        }
                    )

            self.mapping_array = sorted(self.mapping_array, key = lambda el: (el['tbs'], el['mcs']))
            self.action_array = [np.array([x['mcs'], x['prb']]) for x in self.mapping_array] # sort by tbs/mcs
            
            self.tbs_array = []
            self.tbs_to_action_array = []

            tbs = None
            tbs_list = None
            for action_idx in range(len(self.mapping_array)):
                x = self.mapping_array[action_idx]
                x_tbs = x['tbs']
                if (tbs is None or x_tbs > tbs):
                    # got to a new TBS area
                    tbs = x_tbs
                    self.tbs_array.append(tbs)
                    tbs_list = []
                    self.tbs_to_action_array.append(tbs_list)
                tbs_list.append(np.array([action_idx, x['prb'], x['mcs']]))
            self.tbs_len = len(self.tbs_array)
            
            self.action_space = spaces.Discrete(len(self.action_array))
            self.fn_action_translation = self.fn_mcs_prb_joint_action_translation
            self.fn_calculate_mean = self.fn_mcs_prb_joint_mean_calculation
        else:
            raise Exception("Not supported policy output format: " + str(self.policy_output_format))

    def presetup(self, inputs):
        pass

    def setup(self, agent_idx, total_agents):
        self.set_title('worker_{}'.format(agent_idx))

    def get_state_size(self):
        return self.observation_shape[0]

    def get_action_space(self):
        return self.action_space

    def set_title(self, title):
        self.title = title

    def get_title(self):
        return self.title


    def set_observation(self, observation):
        self.observation = observation
    
    def get_observation(self):
        return self.observation

    def get_tbs_array(self):
        return self.tbs_array

    def find_cross_over(self, arr, low, high, x):
        if (arr[high] <= x):
            return high
        
        if (arr[low] > x):
            return low

        mid = (low + high) // 2
        if (arr[mid] <= x and arr[mid + 1] > x):
            return mid
        
        if (arr[mid] < x):
            return self.find_cross_over(arr, mid + 1, high, x)
        
        return self.find_cross_over(arr, low, mid - 1, x)

    def get_closest_actions(self, tbs_hat):
        tbs_array_idx = self.find_cross_over(self.tbs_array, 0, self.tbs_len - 1, tbs_hat)
        tbs_target = self.tbs_array[tbs_array_idx]

        action_index = int(self.tbs_to_action_array[tbs_array_idx][0][0]) # get the one with the lowest PRB
        
        return action_index, tbs_target


    def get_k_closest_actions(self, k, target_tbs):
        solution = self.find_cross_over(self.tbs_array, 0, self.tbs_len - 1, target_tbs)
        tbs_solution = self.tbs_array[solution]
        while (solution > 0 and self.tbs_array[solution - 1] == tbs_solution):
            solution -= 1
        return [solution, self.tbs_array[solution]]

    def get_csv_result_policy_output_columns(self) -> list:
        if (self.policy_output_format == "mcs_prb_joint"):
            columns = ['reward_mean', 'mu_mean', 'sigma_mean', 'crc_ok', 
                    'dec_time_ok_ratio', 'dec_time_ok_mean', 'dec_time_ok_std', 'dec_time_ko_mean', 'dec_time_ko_std', 
                    'throughput_ok_mean', 'throughput_ok_std', 'snr', 'cpu']
            return columns
        else: raise Exception("Not supported policy output format")

    def get_csv_result_policy_output(self, infos) -> list:
        if (self.policy_output_format == "mcs_prb_joint"):
            num_crc_ok = 0
            num_dec_time_ok = 0
            dec_times_ok = []
            dec_times_ko = []
            len_infos = len(infos)
            throughput_ok = []
            mu = []
            sigma = []
            snrs = []
            rewards = []
            cpus = []
            for info in infos:
                crc = info['crc']
                dec_time = info['decoding_time']
                tbs = info['tbs']
                snr = info['snr']
                cpu = info['cpu']
                reward = info['reward']
                if crc:
                    num_crc_ok += 1
                if dec_time < self.decode_deadline:
                    num_dec_time_ok += 1
                    dec_times_ok.append(dec_time)
                else:
                    dec_times_ko.append(dec_time)

                throughput_ok.append(crc * tbs * (dec_time < self.decode_deadline))
                snrs.append(snr)
                cpus.append(cpu)
                if (reward is not None and reward != ''):
                    rewards.append(reward)
                if 'mu' in info:
                    mu.append(info['mu'])
                if 'sigma' in info:
                    sigma.append(info['sigma'])
            reward_mean = np.mean(rewards) if len(rewards) > 0 else -1 
            mu_mean = np.mean(mu) if len(mu) > 0 else -1
            sigma_mean = np.mean(sigma) if len(mu) > 0 else -1
            crc_ok = num_crc_ok / len_infos if len_infos > 0 else -1
            dec_time_ok_ratio = num_dec_time_ok / len_infos if len_infos > 0 else -1
            dec_time_ok_mean = np.mean(dec_times_ok) if len(dec_times_ok) > 0 else -1
            dec_time_ok_std = np.std(dec_times_ok)   if len(dec_times_ok) > 0 else -1
            dec_time_ko_mean = np.mean(dec_times_ko) if len(dec_times_ko) > 0 else -1
            dec_time_ko_std = np.std(dec_times_ko)   if len(dec_times_ko) > 0 else -1
            throughput_ok_mean = np.mean(throughput_ok) / (1024 * 1024) * 1000 if len(throughput_ok) > 0 else -1
            throughput_ok_std = np.std(throughput_ok)   / (1024 * 1024) * 1000 if len(throughput_ok) > 0 else -1
            snr_mean = np.mean(snrs) if len(snrs) > 0 else -1
            cpu_mean = np.mean(cpus) if len(cpus) > 0 else -1
            return [ { 'period': 1, 'value': np.round(reward_mean, 3) },
                     { 'period': 1, 'value': int(mu_mean) },
                     { 'period': 1, 'value': int(sigma_mean) },
                     { 'period': 1, 'value': np.round(crc_ok, 3) }, 
                     { 'period': 1, 'value': np.round(dec_time_ok_ratio, 3) },
                     { 'period': 1, 'value': int(dec_time_ok_mean)},
                     { 'period': 1, 'value': int(dec_time_ok_std)},
                     { 'period': 1, 'value': int(dec_time_ko_mean)},
                     { 'period': 1, 'value': int(dec_time_ko_std)},
                     { 'period': 1, 'value': np.round(throughput_ok_mean, 3)},
                     { 'period': 1, 'value': np.round(throughput_ok_std , 3)}, 
                     { 'period': 1, 'value': np.round(snr_mean, 3) },
                     { 'period': 1, 'value': np.round(cpu_mean, 3) }
                     ]
        else: raise Exception("Can't handle policy output format")


    def to_tbs(self, mcs, prb):
        tbs = 0
        if (prb > 0):
            i_tbs = I_MCS_TO_I_TBS[mcs]
            tbs = self.tbs_table[i_tbs][prb - 1]
        return tbs 

    def get_reward(self, mcs, prb, crc, decoding_time, tbs = None):
        reward = 0
        tbs = None
        if ( prb > 0 and (I_MCS_TO_I_TBS[mcs], prb - 1) in PROHIBITED_COMBOS):
            reward = -1 * self.min_penalty
            return reward, None
        else:
            if tbs is None:
                tbs = self.to_tbs(mcs, prb)
            if (not crc or decoding_time > self.decode_deadline):
                reward = -1 * self.min_penalty
            else:
                reward = (tbs / ( 8 * 1024))
        return reward, tbs

    def get_agent_result(self, reward, mcs, prb, crc, decoding_time, tbs, snr, cpu):
        info = {'mcs': mcs, 'prb': prb, 
                'crc': crc, 'decoding_time': decoding_time, 
                'tbs': tbs, 'snr': snr, 'reward': reward, 'cpu': cpu}
        return None, reward, True, info

    def fn_mcs_prb_joint_action_translation(self, action) -> tuple:
        # in this case action is [action_idx]
        action_idx = action[0]
        assert action_idx >= 0 and action_idx < len(self.action_array), 'Action {} not in range'.format(action_idx)
        mcs, prb = self.action_array[action_idx]
        return int(mcs), int(prb)

    def fn_mcs_prb_joint_mean_calculation(self, probs, info):
        mcs_mean = 0
        prb_mean = 0
        if (probs is not None):
            for prob, action in zip(probs[0], self.action_array):
                mcs_mean += prob * action[0]
                prb_mean += prob * action[1]
            mcs_mean = mcs_mean.numpy()
            prb_mean = prb_mean.numpy()
        else:
            if (len(info) > 0):
                mcs = 0
                prb = 0
                for __key_info in info:
                    mcs += __key_info['mcs']
                    prb += __key_info['prb']
                mcs_mean = mcs / len(info)
                prb_mean = prb / len(info)
        return mcs_mean, prb_mean        

    def translate_action(self, action) -> tuple:
        if (action == 'random'):
            mcs, prb = self.action_array[np.random.randint(0, len(self.action_array))]
            return int(mcs), int(prb)
        else: return self.fn_action_translation(action)

    def calculate_mean(self, probs, info) -> tuple:
        return self.fn_calculate_mean(probs, info)

    def is_state_valid(self) -> bool:
        cpu, snr = self.observation
        is_valid = (cpu >= 0 and cpu <= 3000)
        is_valid = is_valid & (snr >= 0 and snr <=35)
        return is_valid

    def __str__(self) -> str:
        return self.title