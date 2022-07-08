import numpy as np
from multiprocessing import shared_memory


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