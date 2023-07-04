## Utility Functions
- **fct log(text, array=None)**: prints text msg, and optionally if a Numpy array is provided it prints its shape, min and max values.

- **class BatchNorm(KL.BatchNormalization)**: Its purpose seems to be primarily organizational, providing a central place to make modifications if necessary in the future.

- **compute_backbone_shapes(config, image_shape)**: ceil(shape/stride)

## Resnet Graph
- **identity block**: conv(1,1) -> BN -> ReLU -> conv(3,3) -> BN -> ReLU -> conv(1,1) -> BN -> add -> ReLU
- **conv block (uses stride)**: conv(1,1) -> BN -> ReLU -> conv(3,3) -> BN -> ReLU -> conv(1,1) -> BN  || conv(1,1) -> batchnorm || add -> ReLU