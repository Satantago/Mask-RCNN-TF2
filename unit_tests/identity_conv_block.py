import random
import numpy as np
import tensorflow as tf
import keras

from mrcnn.model import conv_block

from tensorflow import shape

# 3D image
data_format = "channels_first"
x = np.random.random((5, 32, 32, 32, 3))
x = tf.convert_to_tensor(x, dtype=tf.float32)
print("x is equal to ", tf.shape(x))

y = conv_block(x, 3, [1,1,1], block='test', stage=1)
print("y conv_block is equal to ", tf.shape(y))

#same for identity block
y = conv_block(x, 3, [1,1,1], block='test', stage=1, use_bias=False, train_bn=False)
print("y identity_block is equal to ", shape(y))