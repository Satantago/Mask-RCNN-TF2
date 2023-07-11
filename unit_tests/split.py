import tensorflow as tf
import numpy as np

# To understand what split does, see this example:
boxes = tf.constant([[0, 0, 1, 1, 2, 2],
                        [3, 3, 4, 4, 5, 5],
                        [6, 6, 7, 7, 8, 8],
                        [9, 9, 10, 10, 11, 11],
                        [12, 12, 13, 13, 14, 14],
                        [10, 10, 11, 11, 12, 12]])  # if it was 5 boxes, it wouldn't work
# as np array please
boxes = boxes.numpy()
boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
x1, y1, z1, x2, y2, z2 = tf.split(boxes, 6, axis=0)  # x1 is first row, y1 is second, ...
print(x1)                                            # axis=1 is columns
print(y1)
print(z1)



# To understand what split in ROIAlign does, see this example:
batch_size = 2
num_boxes = 3

# Generate random values for the sample tensor
sample = np.random.rand(batch_size, num_boxes, 6)

print(sample)
# Convert the numpy array to a TensorFlow tensor
sample = tf.constant(sample, dtype=tf.float32)

# Accessing the individual components
y1, x1, z1, y2, x2, z2 = tf.split(sample, 6, axis=2)

print("y1 is", y1.numpy())
print("x1 is", x1.numpy())
print("z1 is", z1.numpy())
print("y2 is", y2.numpy())
print("x2 is", x2.numpy())
print("z2 is", z2.numpy())