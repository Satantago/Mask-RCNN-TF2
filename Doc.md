## Utility Functions
- **fct log(text, array=None)**: prints text msg, and optionally if a Numpy array is provided it prints its shape, min and max values.

- **class BatchNorm(KL.BatchNormalization)**: Its purpose seems to be primarily organizational, providing a central place to make modifications if necessary in the future.

- **compute_backbone_shapes(config, image_shape)**: ceil(shape/stride)

## Resnet Graph
- **identity block (TO_UPDATE)**:  
conv(1,1) -> BN -> ReLU -> conv(3,3) -> BN -> ReLU -> conv(1,1) -> BN -> add -> ReLU

- **conv block(TO_UPDATE) (uses stride)**: 
conv(1,1) -> BN -> ReLU -> conv(3,3) -> BN -> ReLU -> conv(1,1) -> BN  || conv(1,1) -> batchnorm || add -> ReLU

- **ResNet Graph**:
uses identity and conv blocks to build the ResNet graph. nothing special here. to convert to 3D, modify identity and conv blocks to use 3D convolutions.

## Proposal Layer
- **apply_box_deltas_graph(boxes, deltas)(TO_UPDATE)**: You have MANY boxes.
applies the given deltas to the given boxes. boxes are expected to be in [y1, x1, y2, x2] format. Centers are shifted and then the height and width adjusted.

- **clip_boxes_graph(boxes, window)(TO_UPDATE)**:
 The purpose of this operation is to ensure that the bounding boxes stay within the bounds defined by the window.


## Proposal Layer (TO_UPDATE COMMENT and WINDOW and PADDING IN NMS and ..)
- Input: the rpn_probs tensor provides probabilities for each anchor to be classified as background or foreground, rpn_bbox tensor provides predicted refinements for each anchor's position and size, and anchors tensor contains the coordinates of the anchors in normalized form. These tensors serve as inputs to the ProposalLayer, which processes them to generate a subset of high-scoring bounding box proposals for further stages of object detection.
- The Proposal Layer receives these tensors from Region proposal network and applies bounding box refinements, performs NMS, and generates a fixed number of high-scoring bounding box proposals for further processing in the object detection pipeline.
- In the compute_output_shape method of the ProposalLayer, the returned batch size is None because the batch size is not explicitly specified. The None value is used to indicate that the batch size can vary and is flexible.



## PyramidROIAlign (TO MODIFY METICULOUSLY)
takes bounding box coordinates, image metadata, and feature maps from different levels of the feature pyramid. It performs ROI pooling on each level and outputs pooled regions in a specific shape. This layer is commonly used in object detection models to align features from different pyramid levels with corresponding bounding boxes for further processing and analysis.

## Detection Target Layer

- **overlaps_graph(boxes1, boxes2) (TO_UPDATE)**: Computes IoU overlaps between two sets of boxes.

- **detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config)(TO_UPDATE)**:
generates detection targets for an image given the proposals, ground truth class IDs, ground truth boxes, ground truth masks, and configuration. **This needs to maaaassively change**

- **Class detection target layer**:
serves as a wrapper around the detection_targets_graph function, allowing it to be used as a layer in the model. It performs the necessary computations to generate the detection targets for each proposal in the batch, including ROIs, class IDs, bounding box refinements, and masks.

## Detection Layer

- **refine_detections_graph(rois, probs, deltas, window, config)**: refines classified proposals, filters overlaps, and returns the final detections. 

- **class DetectionLayer(KL.Layer)**: serves as a wrapper around the refine_detections_graph function, allowing it to be used as a layer in the model. It performs the necessary computations to generate the final detections for each proposal in the batch, including bounding box refinements, filtering out overlaps, and applying NMS.

















## ana m3a rasi
- R-CNN and Fast R-CNN use selective search process which is slow

- Faster R-CNN uses Region Proposal Network (RPN) to generate proposals

- RoAlign has to do with converting any H*W feature map into a fixed size feature map. It is a generalization of RoI pooling.

 - we use different anchor boxes to detect all kind of objects (objects that are far away are smaller than objects that are close to the camera)

 - task of RPN is to predict foreground and background anchor boxes and anchor boxes which are labeled as foreground class will go to the next stage

 - RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. RPN is a sub-CNN which means uses some convolutions, so output is a feature map 

 - ROI pooling reduces output feature maps of RPN to a same fixed size. It takes feature map, flattens it 

- 9 anchor boxes are used in Faster R-CNN (3 scales and 3 ratios)
why? because we want to detect objects of different sizes and shapes
what is scale nd ratio? scale is the size of the anchor box and ratio is the ratio of width to height of the anchor box. conv1*1, 1*9 filters because we have 9 anchor boxes and each anchor box has 1 value (objectness score). conv1*1, 4*9 filters because we have 9 anchor boxes and each anchor box has 4 values (bounding box coordinates)

- output after ROI pooling is 7x7x512. Then Fully connected layer is applied to this output. Then 2 branches are created. One branch is for classification and the other is for regression. Classification branch has 2 nodes (0 or 1) and regression branch has 4 nodes (x1, y1, x2, y2).

- if many classes, independtly carry out non max suppression for each class, one on each classes

-  1X1 Conv was used to reduce the number of channels while introducing non-linearity. It can be used for Dimensionality Reduction/Augmentation, Building DEEPER Network and 

- RoIs are predicted not in the original image space but in feature space which is extracted from a vision model


**Anchors**:
In most object detection frameworks, the number of anchors (num_anchors) is fixed and the same for every image in the batch.

The anchors are typically generated based on predefined scales and aspect ratios, which are determined before training the model. These scales and aspect ratios are often chosen to cover a range of object sizes and shapes that are likely to appear in the images.

During training, a set of anchors is generated for each spatial location in the feature map produced by the backbone network. The number of anchors generated at each spatial location is typically determined by the number of predefined scales and aspect ratios.

Since the number of anchors is fixed and determined by the network architecture and anchor settings, it remains the same for all images in a batch. This allows for efficient parallel processing during training and inference, as the network expects a consistent number of anchors for each image.

It's important to note that different object detection models or implementations may use different anchor configurations, and there are variations that allow for dynamically adjusting the number of anchors based on the input image size or other factors. However, in the majority of cases, the number of anchors is fixed and consistent across the images in a batch.


## Faster R-CNN Paper
- Goal is to share computation between RPN and Fast R-CNN
1. small network takes as input an n × n spatial window of the input convolutional feature map.
2. This architecture is naturally implemented with an n × n convolutional layer followed by two sibling 1 × 1 convolutional layers (for reg and cls, respectively).