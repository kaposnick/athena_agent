from tokenize import Number
import numpy as np
import gym
from gym import spaces

NOISE_MIN = -15.0
NOISE_MAX = 100.0
BETA_MIN  = 1.0
BETA_MAX  = 1000.0
BSR_MIN   = 0
BSR_MAX   = 180e3

PROHIBITED_COMBOS = [(0, 0), (0, 1), (0,2), 
                  (1, 0), (1, 1),
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
MCS_SPACE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ,24],  dtype=np.float16)  

I_MCS_TO_I_TBS = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26])


class BaseEnv(gym.Env):
    def __init__(self, 
                input_dims: Number, 
                penalty: Number,
                policy_output_format: str,
                title: str, 
                verbose: Number,
                decode_deadline = 3000) -> None:
        super(BaseEnv, self).__init__()
        self.penalty = penalty
        self.policy_output_format = policy_output_format
        self.title = title
        self.verbose = verbose
        self.decode_deadline = decode_deadline
        self.input_dims = input_dims
        self.observation_shape = (input_dims, )
        self.observation_space = spaces.Box(
            low = np.zeros(self.observation_shape),
            high = np.zeros(self.observation_shape),
            dtype = np.float32
        )

        if (self.policy_output_format == "mcs_prb_joint"):
            self.action_array = []
            self.action_array.append( np.array( [0, 0] ) )
            for mcs in MCS_SPACE:
                for prb in PRB_SPACE:
                    combo = ( I_MCS_TO_I_TBS[int(mcs)], int(prb) - 1)
                    if combo in PROHIBITED_COMBOS:
                        continue
                    self.action_array.append( np.array( [mcs, prb] ) )
            self.action_space = spaces.Discrete(len(self.action_array))
            self.fn_action_translation = self.fn_mcs_prb_joint_action_translation
            self.fn_calculate_mean = self.fn_mcs_prb_joint_mean_calculation

        elif (self.policy_output_format == "mcs_prb_independent"):
            n_actions = (len(MCS_SPACE), len(PRB_SPACE))
            self.action_space = spaces.MultiDiscrete(n_actions)
            self.fn_action_translation = self.fn_mcs_prb_indpendent_action_translation
            self.fn_calculate_mean = self.fn_mcs_prb_independent_mean_calculation
        else:
            raise Exception("Not allowed policy output format: " + str(self.policy_output_format))

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

    def get_csv_result_policy_output_columns(self) -> list:
        if (self.policy_output_format == "mcs_prb_joint"):
            columns = ['crc_ok', 'dec_time_ok', 'dec_time_mean', 'dec_time_std', 'mcs_prb']
            return columns
        elif (self.policy_output_format == "mcs_prb_independent"):
            columns = ['mcs', 'prb']
            return columns
        else: raise Exception("Can't handle policy output format")

    def get_csv_result_policy_output(self, probs, infos) -> list:
        if (self.policy_output_format == "mcs_prb_joint"):
            action_probs = probs[0]
            mcs_prb = str(action_probs.numpy()).replace('\n','')

            num_crc_ok = 0
            num_dec_time_ok = 0
            dec_times = []
            len_infos = len(infos)
            for info in infos:
                crc = info['crc']
                dec_time = info['decoding_time']
                if crc:
                    num_crc_ok += 1
                if dec_time < self.decode_deadline:
                    num_dec_time_ok += 1
                dec_times.append(dec_time)
            crc_ok = num_crc_ok / len_infos if len_infos > 0 else -1
            dec_time_ok = num_dec_time_ok / len_infos if len_infos > 0 else -1
            dec_time_mean = np.mean(dec_times)
            dec_time_std = np.std(dec_times)
            return [ { 'period': 1, 'value': np.round(crc_ok, 3) }, 
                     { 'period': 1, 'value': np.round(dec_time_ok, 3) },
                     { 'period': 1, 'value': int(dec_time_mean)},
                     { 'period': 1, 'value': int(dec_time_std)},
                     { 'period': 100, 'value': mcs_prb }  ]
        elif (self.policy_output_format == "mcs_prb_independent"):
            mcs_probs = probs[0]
            mcs_result = [action_prob.numpy() for action_prob in mcs_probs]
            prb_probs = probs[1]
            prb_result = [action_prob.numpy() for action_prob in prb_probs]
            return mcs_result + prb_result
        else: raise Exception("Can't handle policy output format")


    def to_tbs(self, mcs, prb):
        tbs = 0
        if (prb > 0):
            i_tbs = I_MCS_TO_I_TBS[mcs]
            tbs = self.tbs_table[i_tbs][prb - 1]
        return tbs 

    def get_reward(self, mcs, prb, crc, decoding_time, tbs = None):
        reward = 0
        if ( prb > 0 and (I_MCS_TO_I_TBS[mcs], prb - 1) in PROHIBITED_COMBOS):
            reward = -1 * self.penalty
        else:
            if (not crc):
                reward = - 5
            else:
                if (decoding_time > self.decode_deadline):
                    reward += max((self.decode_deadline - decoding_time) / 100, -5)
                if (tbs is None):
                    tbs = self.to_tbs(mcs, prb) # in KBs
                reward += (tbs / ( 8 * 1024))
        return reward

    def get_agent_result(self, reward, mcs, prb, crc, decoding_time):
        return None, reward, True, {'mcs': mcs, 'prb': prb, 'crc': crc, 'decoding_time': decoding_time}

    def fn_mcs_prb_joint_action_translation(self, action) -> tuple:
        # in this case action is [action_idx]
        action_idx = action[0]
        assert action_idx >= 0 and action_idx < len(self.action_array), 'Action {} not in range'.format(action_idx)
        mcs, prb = self.action_array[action_idx]
        return int(mcs), int(prb)

    def fn_mcs_prb_joint_mean_calculation(self, probs):
        mcs_mean = 0
        prb_mean = 0
        for prob, action in zip(probs[0], self.action_array):
            mcs_mean += prob * action[0]
            prb_mean += prob * action[1]
        return mcs_mean.numpy(), prb_mean.numpy()

    def fn_mcs_prb_indpendent_action_translation(self, action) -> tuple:
        # in this case action is [mcs_action_idx, prb_action_idx]
        action_mcs = action[0]
        action_prb = action[1]
        assert action_mcs >= 0 and action_mcs < len(MCS_SPACE), 'Action MCS {} not in range'.format(action_mcs)
        assert action_prb >= 0 and action_prb < len(PRB_SPACE), 'Action PRB {} not in range'.format(action_prb)
        return int(MCS_SPACE[action_mcs]), int(PRB_SPACE[action_prb])

    def fn_mcs_prb_independent_mean_calculation(self, probs) -> tuple:
        mcs_probs = probs[0]
        prb_probs = probs[1]
        mcs_mean = mcs_probs * MCS_SPACE
        prb_mean = prb_probs * PRB_SPACE
        return mcs_mean, prb_mean

        

    def translate_action(self, action) -> tuple:
        return self.fn_action_translation(action)

    def calculate_mean(self, probs) -> tuple:
        return self.fn_calculate_mean(probs)

    def __str__(self) -> str:
        return self.title