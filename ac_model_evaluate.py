from agents.BaseAgent import BaseAgent
from agents.Config import Config
from common_utils import import_tensorflow

tf, _ = import_tensorflow('3')

config = Config()
config.hyperparameters = {
    'Actor_Critic_Common': {
        'learning_rate': 1e-4,
        'discount_rate': 0.99,
        'linear_hidden_units': [5, 32, 64, 100],
        'num_actor_outputs': 1,
        'final_layer_activation': ['softmax', 'softmax', None],
        'normalise_rewards': False,
        'add_extra_noise': False,
        'tau': 0.005,
        'batch_size': 64,
        'include_entropy_term': True,
        'local_update_period': 1, # in episodes
        'entropy_beta': 0.1,
        'entropy_contrib_prob': 1,
        'Actor': {
            'linear_hidden_units': [100, 40]
        },
        'Critic': {
            'linear_hidden_units': [16, 4]
        }
    },
        
}


model1 = BaseAgent.create_NN(tf, 3, [564, 1], config.hyperparameters)
model1.load_weights('/home/naposto/phd/nokia/data/csv_45/low_noise_high_beta_joint/run_0_beta_high_noise_low_entropy_0.1.h5')

noise_range = (-15, 50)
beta_range = (650, 700)

import numpy as np
noise = np.random.uniform(low = noise_range[0], high = noise_range[1], size = (1024, 1))
beta = np.random.uniform(low = beta_range[0], high = beta_range[1], size = (1024, 1))

input_tensor = tf.concat([noise_range, beta_range], axis = 1)

