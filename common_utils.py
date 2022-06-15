
def import_tensorflow(debug_level: str):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
    import tensorflow as tf
    return tf, os
