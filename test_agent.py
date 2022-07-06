from A3CAgent import A3CAgent
from env.DecoderEnv import DecoderEnv
from Config import Config

decoder_model_path = '/home/naposto/phd/nokia/digital_twin/models/ce_na_na_wo_clip'
tbs_table_path    = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json'

beta_low = (1, 50)
beta_medium = (50, 650)
beta_high = (650, 1000)
beta_all = (1, 1000)

noise_low = (-15,50)
noise_medium = (50,80)
noise_high = (80, 100)
noise_all = (-15, 100)

decoder_env = DecoderEnv(
    # decoder_model_path = decoder_model_path,
    # tbs_table_path     = tbs_table_path,
    decoder_model_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/crc_ce_dcd_time_prob.h5',
    input_norm_mean_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_mean.npy',
    input_norm_var_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_var.npy',
    tbs_table_path = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json',
    noise_range        = noise_high,
    beta_range         = beta_high
)

config = Config()
config.seed = 1
config.environment = decoder_env
config.num_episodes_to_run = 4
config.randomise_random_seed = False
config.runs_per_agent = 1
config.save_results = True
config.save_weights = True
config.load_initial_weights = False
config.save_weights_period = 100
config.hyperparameters = {
    'Actor_Critic_Common': {
        'learning_rate': 1e-4,
        'linear_hidden_units': [5, 32, 64, 100],
        'num_actor_outputs': 1,
        'use_state_value_critic': False,
        'final_layer_activation': ['softmax'],
        'batch_size': 64,
        'include_entropy_term': True,
        'local_update_period': 1, # in episodes
        'entropy_beta': 0.1,
        'entropy_contrib_prob': 1,
        'Actor': {
            'linear_hidden_units': [100, 40]
        },
        'State_Value_Critic': {
            'linear_hidden_units': [16, 4]
        },
        'Action_Value_Critic': {
            'linear_hidden_units': [16, 32, 32],
            'final_layer_activation': 'softmax',
            'vmin': -5, 
            'vmax': 4,
            'n_atoms': 20 
        }
    },
        
}

import copy
for beta_range, beta in zip([beta_high], ['all']):
    for noise_range, noise in zip([noise_low], ['all']):
        decoder_env = DecoderEnv(
            decoder_model_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/crc_ce_dcd_time_prob.h5',
            input_norm_mean_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_mean.npy',
            input_norm_var_path = '/home/naposto/phd/nokia/digital_twin/models/new_model/scaler_var.npy',
            tbs_table_path = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json',
            noise_range = noise_range, 
            beta_range= beta_range,
            policy_output_format = 'mcs_prb_joint',
            version='probabilistic',
            sample_strategy = 'percentile_99',
            input_dims=2
        )
        for run_idx in range(config.runs_per_agent):
            seed = run_idx * 32 + 1
            config = copy.deepcopy(config)
            config.seed = seed
            config.environment = decoder_env
            # save_folder = '/home/naposto/phd/nokia/agent_models/model_v2/model_training_output.csv'
            # config.results_file_path = save_folder.format(run_idx)
            config.results_file_path = '/tmp/dt_twin.csv'
            config.save_weights = False
            # config.save_weights_file = '/home/naposto/phd/nokia/agent_models/model_v2/model_weights.h5'.format(run_idx)
            A3C_Agent = A3CAgent(config, 1)
            A3C_Agent.run_n_episodes()