import tensorflow as tf
BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DEPTH = 256
CHANNELS = 3
CROP_SIZE = (24, 24)

# image = .ratfndom.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, CHANNELS))
# boxes = tf.random.uniform(shape=(NUM_BOXES, 6))
# box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
# maxval=BATCH_SIZE, dtype=tf.int32)
# output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
# print(output.shape)




# reshaped_tensor = tf.reshape(your_3d_tensor, [batch_size * depth, height, width, channels])
# cropped_resized_tensor = tf.image.crop_and_resize(reshaped_tensor, boxes, box_indices, crop_size)
# result_tensor = tf.reshape(cropped_resized_tensor, [batch_size, depth, output_height, output_width, channels])




import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import mrcnn.utils as utils


# Load the 3D image
image_path = 'sample.nii.gz'  # Replace with the path to your 3D image file
image = sitk.ReadImage(image_path)
# print(image.GetSize())
# image =  image[:,:,59]
np_array = sitk.GetArrayFromImage(image)
print(np_array.shape)

cropped_and_resized = utils.crop_resize_one_image(np_array, [0.4, 0.4, 0.4, 0.7, 0.7, 0.7], [2,2,2])
cropped_and_resized = cropped_and_resized[:,:,0]
plt.imshow(cropped_and_resized)
plt.show()
