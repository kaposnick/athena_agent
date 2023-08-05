import numpy as np
from multiprocessing import shared_memory

MODE_SCHEDULING_SRS = 0
MODE_SCHEDULING_ATHENA = 1
MODE_SCHEDULING_RANDOM = 2

MODE_TRAINING = 1
MODE_INFERENCE = 0

PROHIBITED_COMBOS = [(0, 0), (0, 1), (0,2), (0, 3), 
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1),
                  (3, 0), 
                  (4, 0), 
                  (5, 0), 
                  (6, 0)]

PRB_SPACE = np.array([1, 2, 3, 4, 5, 6, 8, 9, 
                    10, 12, 15, 16, 18, 
                    20, 24, 25, 27, 
                    30, 32, 36, 40, 45], dtype = np.float16)

MCS_SPACE =      np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  dtype=np.float16)
I_MCS_TO_I_TBS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26])

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
        try:
            shm = shared_memory.SharedMemory(name=share_memory_name, create=True, size=size)        
        except:
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
 
tbs_table_path = 'resources/cpp_tbs.json'
import json
with open(tbs_table_path) as tbs_json:
    tbs_table = json.load(tbs_json)

def to_tbs(mcs, prb):
    tbs = 0
    if (prb > 0):
        i_tbs = I_MCS_TO_I_TBS[mcs]
        tbs = tbs_table[i_tbs][prb - 1]
    return tbs


def get_action_array():
    mapping_array = []
    for mcs in MCS_SPACE:
        for prb in PRB_SPACE:
            combo = ( I_MCS_TO_I_TBS[int(mcs)], int(prb) - 1)
            if combo in PROHIBITED_COMBOS:
                continue
            mapping_array.append(
                {   
                    'tbs': to_tbs(int(mcs), int(prb)),
                    'mcs': mcs,
                    'prb': prb
                }
            )

    mapping_array = sorted(mapping_array, key = lambda el: (el['tbs'], el['mcs']))
    action_array = [np.array([x['mcs'], x['prb']]) for x in mapping_array] # sort by tbs/mcs
    action_array = np.array(action_array)
    return action_array





