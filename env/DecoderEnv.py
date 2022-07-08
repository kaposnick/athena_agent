import json
from tokenize import Number
import numpy as np
from env.BaseEnv import NOISE_MIN, NOISE_MAX, BETA_MIN, BETA_MAX, BSR_MIN, BSR_MAX

from common_utils import import_tensorflow
from env.BaseEnv import BaseEnv

class DecoderEnv(BaseEnv):
    def __init__(self,
                 decoder_model_path,
                 tbs_table_path,
                 version='probabilistic',
                 sample_strategy = 'percentile_99',
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
            self.sample_strategy = sample_strategy

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
        self.tf, _, _ = import_tensorflow('3')
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
            if (self.sample_strategy == 'sample'):
                # sampling 
                decoding_time = self.tfd.Normal(loc = dcd_time_mean, scale = dcd_time_stdv).sample()[0][0].numpy()
            elif (self.sample_strategy == 'percentile_99'):
                # 99% percentile -> Z = 2.326
                decoding_time = (dcd_time_mean + 2.326 * dcd_time_stdv)[0][0].numpy()
        else:
            decoding_time = decoding_time_output

        crc = np.random.binomial(n = 1, p = decode_prob)[0][0] 

        reward, tbs = super().get_reward(mcs, prb, crc, decoding_time)
        result = super().get_agent_result(reward, mcs, prb, crc, decoding_time, tbs)

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
            beta_range= [650, 1000],
            policy_output_format = 'mcs_prb_joint'
    )
    environment.setup(0, 1)
    environment.reset()
    environment.step([48])
    