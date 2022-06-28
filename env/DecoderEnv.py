import json
from tokenize import Number
import gym

from gym import spaces
from common_utils import import_tensorflow
import numpy as np

NOISE_MIN = -15.0
NOISE_MAX = 100.0
BETA_MIN  = 1.0
BETA_MAX  = 700.0
BSR_MIN   = 0
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
                            14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26])

class BaseEnv(gym.Env):
    def __init__(self, 
                input_dims: Number, 
                penalty: Number,
                policy_output_format: str,
                title: str, 
                verbose: Number) -> None:
        super(BaseEnv, self).__init__()
        self.penalty = penalty
        self.policy_output_format = policy_output_format
        self.title = title
        self.verbose = verbose
        self.input_dims = input_dims
        self.observation_shape = (input_dims, )
        self.observation_space = spaces.Box(
            low = np.zeros(self.observation_shape),
            high = np.zeros(self.observation_shape),
            dtype = np.float32
        )

        if (self.policy_output_format == "mcs_prb_joint"):
            self.action_array = []
            for mcs in MCS_SPACE:
                for prb in PRB_SPACE:
                    if (mcs, prb) in PROHIBITED_COMBOS:
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
            columns = ['mcs_prb']
            return columns
        elif (self.policy_output_format == "mcs_prb_independent"):
            columns = ['mcs', 'prb']
            return columns
        else: raise Exception("Can't handle policy output format")

    def get_csv_result_policy_output(self, probs) -> list:
        if (self.policy_output_format == "mcs_prb_joint"):
            action_probs = probs[0]
            result = [action_prob.numpy() for action_prob in action_probs]
            return result
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
        if ( self.policy_output_format == "mcs_prb_independent" and (mcs, prb) in PROHIBITED_COMBOS):
            reward = -1 * self.penalty
        else:
            if (crc == True and decoding_time <= 3000):
                if (tbs is None):
                    tbs = self.to_tbs(mcs, prb) # in KBs
                reward = tbs / ( 8 * 1024)
            else:
                reward = -1 * self.penalty
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
        for prob, action in zip(probs[0][0], self.action_array):
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

class DecoderEnv(BaseEnv):
    def __init__(self,
                 decoder_model_path,
                 tbs_table_path,
                 version='probabilistic',
                 input_norm_mean_path = None,
                 input_norm_var_path = None,
                 noise_range = [NOISE_MIN, NOISE_MAX], 
                 beta_range  = [BETA_MIN,  BETA_MAX],
                 bsr_range   = [BSR_MIN,   BSR_MAX],
                 input_dims = 3,
                 penalty = 15,
                 policy_output_format = "mcs_prb_independent",
                 title = 'Digital Twin Environment',
                 verbose = 0) -> None:
        super(DecoderEnv, self).__init__(
            input_dims = input_dims,
            penalty = penalty,
            policy_output_format = policy_output_format,
            title = title, 
            verbose = verbose)
        self.decoder_model_path = decoder_model_path
        self.tbs_table_path = tbs_table_path

        assert version in ['probabilistic', 'deterministic'], 'Version {} unknown format'.format(version)
        self.version = version
        if (version == 'probabilistic'):
            self.input_norm_mean_path = input_norm_mean_path
            self.input_norm_var_path  = input_norm_var_path

        if noise_range is not None:
            assert noise_range[0] >= NOISE_MIN and noise_range[1] <= NOISE_MAX
        self.noise_range = noise_range

        if beta_range is not None:
            assert beta_range[0] >= BETA_MIN and beta_range[1] <= BETA_MAX
        self.beta_range = beta_range

        if bsr_range is not None:
            assert bsr_range[0] >= BSR_MIN and bsr_range[1] <= BSR_MAX
        self.bsr_range = bsr_range

    def build_model(self, ):
        from tensorflow import keras
        from tensorflow.keras import layers

        # beta, prb, mcs, noise
        inputs = keras.Input(shape=4)
        scaler_mean = np.load(self.input_norm_mean_path)
        scaler_var = np.load(self.input_norm_var_path)
        normalizer = layers.Normalization(mean = scaler_mean, variance = scaler_var)

        x = normalizer(inputs)
        x = layers.Dense(5 , activation='relu', ) (x)
        x = layers.Dense(8 , activation='relu', ) (x)
        x = layers.Dense(20, activation='relu', ) (x)
        x = layers.Dense(40, activation='relu', ) (x)
        x = layers.Dense(40, activation='relu', ) (x)
        x = layers.Dense(40, activation='relu', ) (x)
        x = layers.Dense(40, activation='relu', ) (x)
        x = layers.Dense(100, activation='relu', ) (x)

        crc = layers.Dense(512, activation='relu', ) (x)
        crc = layers.Dense(512, activation='relu', ) (crc)
        crc = layers.Dense(256, activation='relu', ) (crc)
        crc = layers.Dense(100, activation='relu', ) (crc)
        crc = layers.Dense(100, activation='relu', ) (crc)
        crc = layers.Dense(40, activation='relu', ) (crc)

        output_crc = layers.Dense( 1, activation='sigmoid', name = "output_crc") (crc)

        dcd_time = layers.Dense(50, activation='relu') (x)
        dcd_time = layers.Dense(25, activation='relu') (dcd_time)
        output_dcd_time = layers.Dense(1 + 1, activation = 'linear') (dcd_time) # mean, std
        output_dcd_time = self.tfp.layers.DistributionLambda(lambda t: self.tfd.Normal(loc = t[..., :1],
                                                                            scale = 1e-3 + self.tf.math.softplus(0.05 * t[..., 1:])),
                                                        name = "output_time") (output_dcd_time)
        
        model = keras.Model(inputs = inputs, outputs = [output_crc, output_dcd_time])
        model.load_weights(self.decoder_model_path)
        return model


    def setup(self, agent_idx, total_agents):
        super().setup(agent_idx, total_agents)
        self.tf, _ = import_tensorflow('3')
        if (self.version == 'probabilistic'):
            import tensorflow_probability as tfp
            self.tfp = tfp
            self.tfd = tfp.distributions            
            self.decoder_model = self.build_model()
        else:
            self.decoder_model = self.tf.keras.models.load_model(self.decoder_model_path, compile = False)

        with open(self.tbs_table_path) as tbs_json:
            self.tbs_table = json.load(tbs_json)       
        
    def step(self, action):        
        mcs, prb = super().translate_action(action)
        observation = self.get_observation() # noise, beta, bsr
        noise = observation[0]
        beta  = observation[1]

        # digital twin input: beta, prb, mcs, noise
        decoder_input_array = np.array([[beta, prb, mcs, noise]], dtype=np.float32)
        decoder_input = self.tf.convert_to_tensor(decoder_input_array)

        # crc, decoding_time = self.decoder_model.predict(decoder_input, batch_size = 1)
        decode_prob, decoding_time_output = self.decoder_model(decoder_input, training = False)
        if (self.version == 'probabilistic'):
            dcd_time_mean, dcd_time_stdv = decoding_time_output.mean(), decoding_time_output.stddev()
            decoding_time = self.tfd.Normal(loc = dcd_time_mean, scale = dcd_time_stdv).sample()[0][0].numpy()
        else:
            decoding_time = decoding_time_output

        crc = np.random.binomial(n = 1, p = decode_prob)[0][0] 

        reward = super().get_reward(mcs, prb, crc, decoding_time)
        result = super().get_agent_result(reward, mcs, prb, crc, decoding_time)

        if (self.version == 'probabilistic'):
            result[3]['decoding_time_mean'] = dcd_time_mean[0][0].numpy()
            result[3]['decoding_time_stdv'] = dcd_time_stdv[0][0].numpy()

        return result

    def reset(self):
        super().set_observation( 
            [
                np.random.uniform(low = self.noise_range[0], high = self.noise_range[1]),
                np.random.uniform(low = self.beta_range[0] , high = self.beta_range[1] ), 
                np.random.uniform(low = self.bsr_range[0]  , high = self.bsr_range[1]  )
            ][:self.input_dims]
        )
        return super().get_observation()

if (__name__=="__main__"):
    environment = DecoderEnv(
            decoder_model_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/crc_ce_dcd_time_prob.h5',
            input_norm_mean_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_mean.npy',
            input_norm_var_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_var.npy',
            tbs_table_path = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json',
            noise_range = [-15, 50], 
            beta_range= [650, 700],
            policy_output_format = 'mcs_prb_joint'
    )
    environment.setup(0, 1)
    environment.reset()
    environment.step([48])
    