import tensorflow as tf
import numpy as np

# create a random [batch, height, width, depth, channels] tensor
x = tf.random.normal(shape=(16, 256, 256, 256, 3))
# print(x)
depth_tensors = tf.unstack(x, axis=3)
# print(depth_tensors)
for image_stack in depth_tensors:
    # tf.image.crop_and_resize(image_stack, boxes, box_indices, crop_size
    print(image_stack.shape)
print(depth_tensors.shape)