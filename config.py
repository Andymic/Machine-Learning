# @Author: Michel Andy <ragnar>
# @Date:   2020-01-25T10:57:09-05:00
# @Email:  Andymic12@gmail.com
# @Filename: config.py
# @Last modified by:   ragnar
# @Last modified time: 2020-01-25T11:05:09-05:00
import os
import tensorflow as tf

DATA_DIR = os.path.join(os.getcwd(), 'data', '')

# Something weird about TF2 where you need to explicitly set the mem growth for the 
# physical devices otherwise cuda fail to initialize
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Tensorflow GPU config
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = True
SESS = tf.compat.v1.Session(config=tf_config)

# Tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

