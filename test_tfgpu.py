import tensorflow as tf

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

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#  try:
#    tf.config.experimental.set_virtual_device_configuration(
#        gpus[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Virtual devices must be set before GPUs have been initialized
#    print(e)
#
print(tf.matmul([[1., 2.],[3., 4.]], [[1., 2.],[3., 4.]]))
