import json
import gym

from gym import spaces
from common_utils import import_tensorflow
import numpy as np

NOISE_MIN = 10.0
NOISE_MAX = 100.0
BETA_MIN  = 1.0
BETA_MAX  = 700.0
BSR_MIN   = 120e3
BSR_MAX   = 180e3

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
MCS_SPACE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ,24],  dtype=np.float16)  

I_MCS_TO_I_TBS = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26]);

class DecoderEnv(gym.Env):
    def __init__(self, 
                 decoder_model_path,
                 tbs_table_path, 
                 noise_range = None, 
                 beta_range  = None,
                 bsr_range   = None,
                 penalty = 15,
                 title = 'LTE Turbo Decoder') -> None:
        super(DecoderEnv, self).__init__()
        self.decoder_model_path = decoder_model_path
        self.tbs_table_path = tbs_table_path
        self.penalty = penalty
        self.title = title

        if noise_range is not None:
            assert noise_range[0] >= NOISE_MIN and noise_range[1] <= NOISE_MAX
            self.noise_min = noise_range[0]
            self.noise_max = noise_range[1]
        else:
            self.noise_min = NOISE_MIN
            self.noise_max = NOISE_MAX

        if beta_range is not None:
            assert beta_range[0] >= BETA_MIN and beta_range[1] <= BETA_MAX
            self.beta_min = beta_range[0]
            self.beta_max = beta_range[1]
        else:
            self.beta_min = BETA_MIN
            self.beta_max = BETA_MAX

        if bsr_range is not None:
            self.bsr_min  = bsr_range[0]
            self.bsr_max  = bsr_range[1]
        else:
            self.bsr_min  = BSR_MIN
            self.bsr_max  = BSR_MAX

        # Define a 1-D observation space
        self.observation_shape = (3,)
        self.observation_space = spaces.Box(
                            low  = np.zeros(self.observation_shape),
                            high = np.zeros(self.observation_shape), 
                            dtype = np.float32)

        
        
        n_actions = (len(MCS_SPACE), len(PRB_SPACE))
        
        self.action_space = spaces.MultiDiscrete(n_actions)

    def get_environment_title(self):
        return self.title

    def translate_action(self, action):
        return MCS_SPACE[action[0]], PRB_SPACE[action[1]]

    def get_observation(self):
        return self.observation

    def to_tbs(self, mcs, prb):
        tbs = 0
        if (prb > 0):
            i_tbs = I_MCS_TO_I_TBS[mcs]
            tbs = self.tbs_table[i_tbs][prb - 1]
        return tbs

    def reward(self, mcs, prb, crc, decoding_time):
        reward = 0
        if ( (mcs, prb) in PROHIBITED_COMBOS):
            reward = -1 * self.penalty
        else:
            if (crc == True and decoding_time <= 3000):
                reward = self.to_tbs(mcs, prb) / (8 * 1024) # in KBs
            else:
                reward = -1 * self.penalty
        return reward
        
    def step(self, action):
        if (not hasattr(self, 'decoder_model')):
            self.tf, _ = import_tensorflow('3')
            self.decoder_model = self.tf.keras.models.load_model(self.decoder_model_path, compile = False)

        if (not hasattr(self, 'tbs_table')):
            with open(self.tbs_table_path) as tbs_json:
                self.tbs_table = json.load(tbs_json)

        action_mcs = action[0]
        action_prb = action[1]
        assert action_mcs >= 0 and action_mcs < len(MCS_SPACE), 'Action MCS {} not in range'.format(action_mcs)
        assert action_prb >= 0 and action_prb < len(PRB_SPACE), 'Action PRB {} not in range'.format(action_prb)
        
        mcs, prb = self.translate_action(action)
        noise, beta, bsr = self.get_observation()
        decoder_input = self.tf.convert_to_tensor(np.array([[beta, prb, mcs, noise]], dtype=np.float32))

        # crc, decoding_time = self.decoder_model.predict(decoder_input, batch_size = 1)
        decode_prob, decoding_time = self.decoder_model(decoder_input, training = False)
        crc = np.random.binomial(n = 1, p = decode_prob)[0][0] 
        reward = self.reward(int(mcs), int(prb), crc, decoding_time)

        return None, reward, True, {} 

    def reset(self):
        self.observation = (
            np.random.uniform(low = self.noise_min,  high = self.noise_max),
            np.random.uniform(low = self.beta_min ,  high = self.beta_max ), 
            np.random.uniform(low = self.bsr_min  ,  high = self.bsr_max  )
        )
        return self.observation
        