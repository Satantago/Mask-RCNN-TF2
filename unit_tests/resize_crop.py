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

# Load the 3D image
image_path = 'path_to_your_image.nii.gz'  # Replace with the path to your 3D image file
image = sitk.ReadImage(image_path)

# Get the image dimensions
size = image.GetSize()
width, height, depth = size

# Accessing voxel values
voxel_value = image.GetPixel(x, y, z)  # Replace x, y, z with the desired voxel coordinates

# Convert the image to a NumPy array
image_array = sitk.GetArrayFromImage(image)

# Perform operations on the image array
# ...

# Save the modified image
output_path = 'path_to_save_output.nii.gz'  # Replace with the desired output path
output_image = sitk.GetImageFromArray(image_array)
sitk.WriteImage(output_image, output_path)
