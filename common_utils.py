import numpy as np
from multiprocessing import shared_memory

from env.BaseEnv import BETA_MAX, BETA_MIN, BSR_MAX, BSR_MIN, NOISE_MAX, NOISE_MIN


def import_tensorflow(debug_level: str, import_tfp = False):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
    import tensorflow as tf

    tfp = None
    if (import_tfp):
        import tensorflow_probability as tfp
    return tf, os, tfp

def get_shared_memory_ref(
        size, dtype, share_memory_name):
        total_variables = int( size / dtype.itemsize )
        shm = shared_memory.SharedMemory(name=share_memory_name, create=False, size=size)        
        shared_weights_array = np.ndarray(
                                shape = (total_variables, ),
                                dtype = dtype,
                                buffer = shm.buf)
        return shm, shared_weights_array

def map_weights_to_shared_memory_buffer(weights, shared_memory_buf):
        buffer_idx = 0
        for idx_weight in range(len(weights)):
            weight_i = weights[idx_weight]
            shape = weight_i.shape
            size  = weight_i.size
            weights[idx_weight] = np.ndarray(shape = shape,
                                           dtype = weight_i.dtype,
                                           buffer = shared_memory_buf[buffer_idx: (buffer_idx + size)])
            buffer_idx += size

        return weights

def publish_weights_to_shared_memory(weights, shared_ndarray):
        buffer_idx = 0
        for weight in weights:
            flattened = weight.flatten().tolist()
            size = len(flattened)
            shared_ndarray[buffer_idx: (buffer_idx + size)] = flattened
            buffer_idx += size

def save_weights(model, save_weights_file, throw_exception_if_error = False):
    try:
        print('Saving weights to {} ...'.format(save_weights_file), end='')
        model.save_weights(save_weights_file)
        print('Success')
    except Exception as e:
        print('Error' + str(e))
        if (throw_exception_if_error):
            raise e

def get_state_normalization_layer(tf, num_states):
    mean = np.array([NOISE_MIN, BETA_MIN, BSR_MIN], dtype=np.float32)[:num_states]
    variance = np.power(np.array([
        NOISE_MAX - NOISE_MIN,
        BETA_MAX  - BETA_MIN ,
        BSR_MAX   - BSR_MIN
    ], dtype=np.float32), 2)[:num_states]
    normalization_layer = tf.keras.layers.Normalization(
        axis = -1, mean = mean, variance = variance
    )
    return normalization_layer

def get_action_normalization_layer(tf, max_action_idx):
    action_min = 0
    action_max = max_action_idx
    action_variance = np.power(action_max - action_min , 2)

    normalization_layer = tf.keras.layers.Normalization(
        axis = -1, mean = [action_min], variance = [action_variance])
    return normalization_layer
    

def get_basic_actor_network(tf, num_states):
    layers = tf.keras.layers    
    state_normalization_layer = get_state_normalization_layer(tf, num_states)

    inputs = layers.Input(shape = (num_states), name = 'actor_input_layer')
    state_normalization_layer = state_normalization_layer(inputs)
    dense_layer  = layers.Dense(256, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal()) (state_normalization_layer)
    dense_layer  = layers.Dense(256, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal()) (dense_layer)
    output_layer = layers.Dense(1,   activation = 'sigmoid') (dense_layer)
    
    model = tf.keras.Model(inputs, output_layer)
    return model

def get_basic_critic_network(tf, num_states, num_actions, max_action_idx):
    layers = tf.keras.layers
    state_normalization_layer = get_state_normalization_layer(tf, num_states)
    action_normalization_layer = get_action_normalization_layer(tf, max_action_idx)

    state_input = layers.Input(shape = (num_states), name = 'critic_state_input_layer')
    state_normalization_layer = state_normalization_layer(state_input)
    state_out   = layers.Dense(16, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(state_normalization_layer)
    state_out   = layers.Dense(32, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(state_out)
    
    action_input = layers.Input(shape = (num_actions), name = 'critic_action_input_layer')
    action_normalization_layer = action_normalization_layer(action_input)
    action_out   = layers.Dense(32, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(action_normalization_layer)

    concat       = layers.Concatenate()([state_out, action_out])
    out          = layers.Dense(64, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(concat)
    out          = layers.Dense(64, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal())(out)
    outputs      = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model




