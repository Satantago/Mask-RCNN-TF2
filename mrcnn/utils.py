"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import logging
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, depth, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, z1, y2, x2, z2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        depth_indicies = np.where(np.any(m, axis=2))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = depth_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            z2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, z2, y1, y2, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, z1, y2, x2, z2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, z1, y2, x2, z2]
    boxes: [boxes_count, (y1, x1, z1, y2, x2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[4], boxes[:, 4])
    z1 = np.maximum(box[2], boxes[:, 2])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, z1, y2, x2, z2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    area2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, Depth, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, z1, y2, x2, z2)]. Notice that (y2, x2, z2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    z1 = boxes[:, 2]
    y2 = boxes[:, 3]
    x2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (y2 - y1) * (x2 - x1) * (z2 - z1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, z1, y2, x2, z2)]. Note that (y2, x2, z2) is outside the box.
    deltas: [N, (dy, dx, dz, log(dh), log(dw), log(dd))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 3] - boxes[:, 0]
    width = boxes[:, 4] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 2]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 2] + 0.5 * depth
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= np.exp(deltas[:, 3])
    width *= np.exp(deltas[:, 4])
    depth *= np.exp(deltas[:, 5])
    # Convert back to y1, x1, z1, y2, x2, z2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    return np.stack([y1, x1, z1, y2, x2, z2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, z1, y2, x2, z2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 3] - box[:, 0]
    width = box[:, 4] - box[:, 1]
    depth = box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
    center_z = box[:, 2] + 0.5 * depth

    gt_height = gt_box[:, 3] - gt_box[:, 0]
    gt_width = gt_box[:, 4] - gt_box[:, 1]
    gt_depth = gt_box[:, 5] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    gt_center_z = gt_box[:, 2] + 0.5 * gt_depth

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dz = (gt_center_z - center_z) / depth
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)
    dd = tf.math.log(gt_depth / depth)

    result = tf.stack([dy, dx, dz, dh, dw, dd], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, z1, y2, x2, z2)]. (y2, x2, z2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 3] - box[:, 0]
    width = box[:, 4] - box[:, 1]
    depth = box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
    center_z = box[:, 2] + 0.5 * depth

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_depth = gt_box[:, 4] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    gt_center_z = gt_box[:, 2] + 0.5 * gt_depth
    
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dz = (gt_center_z - center_z) / depth
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)
    dd = np.log(gt_depth / depth)

    return np.stack([dy, dx, dz, dh, dw, dd], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,D,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, depth, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim, max_dim].
        pad64: Pads width and height and depth with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, z1, y2, x2, z2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2, z2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (z_left, z_right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, z1, y2, x2, z2) and default scale == 1.
    h, w, d = image.shape[:3]
    window = (0, 0, 0, h, w, d)
    scale = 1
    padding = [(0, 0), (0, 0), (0,0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w, d))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w, d)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), round(d * scale),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width and depth
        h, w, d = image.shape[:3]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        z_left_pad = (max_dim - d) // 2
        z_right_pad = max_dim - d - z_left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (z_left_pad, z_right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, z_left_pad, h + top_pad, w + left_pad, d + z_left_pad)
    elif mode == "pad64":
        h, w, d = image.shape[:3]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        # Depth
        if d % 64 > 0:
            max_d = d - (d % 64) + 64
            z_left_pad = (max_d - d) // 2
            z_right_pad = max_d - d - z_left_pad
        else:
            z_left_pad = z_right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (z_left_pad, z_right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, z_left_pad, h + top_pad, w + left_pad, d + z_left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w, d = image.shape[:3]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        z = random.randint(0, (d - min_dim))
        crop = (y, x, z, min_dim, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim, z:z + min_dim]
        window = (0, 0, 0, min_dim, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (z_left, z_right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, scale, 1], order=0)
    if crop is not None:
        y, x, z, h, w, d = crop
        mask = mask[y:y + h, x:x + w, z:z + d]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        y1, x1, z1, y2, x2, z2 = bbox[i][:6]
        m = m[y1:y2, x1:x2, z1:z2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:3] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, :, i]
        y1, x1, z1, y2, x2, z2 = bbox[i][:6]
        h = y2 - y1
        w = x2 - x1
        d = z2 - z1
        # Resize with bilinear interpolation
        m = resize(m, (h, w, d))
        mask[y1:y2, x1:x2, z1:z2, i] = np.around(m).astype(np.bool)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width, depth] of type float. A small, typically 28x28x28 mask.
    bbox: [y1, x1, z1, y2, x2, z2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, z1, y2, x2, z2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1, z2 - z1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:3], dtype=np.bool)
    full_mask[y1:y2, x1:x2, z1:z2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width, depth] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / (ratios ** (1/3))
    widths = scales / (ratios ** (1/3))
    depths = scales * (ratios ** (2/3))

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_z = np.arange(0, shape[2], anchor_stride) * feature_stride

    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights and depth
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=3).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=3).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, z1, y2, x2, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, z1, y2, x2, z2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, z1, y2, x2, z2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, z1, y2, x2, z2)] in image coordinates
    gt_boxes: [N, (y1, x1, z1, y2, x2, z2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    """
    h, w, d = shape
    scale = np.array([h - 1, w - 1, d - 1, h - 1, w - 1, d - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, z1, y2, x2, z2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2, z2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, z1, y2, x2, z2)] in pixel coordinates
    """
    h, w, d = shape
    scale = np.array([h - 1, w - 1, d - 1,  h - 1, w - 1, d - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)



def nms_3d(bboxes, psocres, threshold, proposal_count):
    bboxes = bboxes.astype('float')
    y_min = bboxes[:,0]
    x_min = bboxes[:,1]
    z_min = bboxes[:,2]
    y_max = bboxes[:,3]
    x_max = bboxes[:,4]
    z_max = bboxes[:,5]
    
    sorted_idx = psocres.argsort()[::-1]
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)*(z_max-z_min+1)
    
    filtered = []
    while len(sorted_idx) > 0:
        rbbox_i = sorted_idx[0]
        filtered.append(rbbox_i)
        
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_zmins = np.maximum(z_min[rbbox_i],z_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        overlap_zmaxs = np.minimum(z_max[rbbox_i],z_max[sorted_idx[1:]])
        
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_depths = np.maximum(0,(overlap_zmaxs-overlap_zmins+1))

        overlap_areas = overlap_widths*overlap_heights*overlap_depths
        
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))

        sorted_idx = np.delete(sorted_idx,delete_idx)

    bboxes = bboxes[filtered].astype('int')
    padding = tf.maximum(proposal_count - tf.shape(bboxes)[0], 0)
    bboxes = np.pad(bboxes, ((0, padding), (0, 0)), 'constant', constant_values=(0, 0))
    return tf.convert_to_tensor(bboxes)




def precalc_bilinear(height, width, depth, crop_size, roi_start_h, roi_start_w, roi_start_d,
                    bin_size_h, bin_size_w, bin_size_d, roi_bin_grid_h, roi_bin_grid_w, roi_bin_grid_d):
    precalc = dict()
    pre_calc_index = 0
    # ph, pw and pd defines each cube that RoI is being divided into. For example,
    # pooled_height = pooled_width = pooled_depth = 2 we divide RoI into 8 volumes thus each ph, pw and dd
    # we sample points from each of those areas (e.g. 0,0,0 - sample from cube 0,0,0)
    pooled_height = crop_size[0]
    pooled_width = crop_size[1]
    pooled_depth = crop_size[2]
    for ph in range(int(pooled_height)):
        for pw in range(int(pooled_width)):
            for pd in range(int(pooled_depth)):
                # iy and ix represent sampled points within each area in RoI. 
                # For example, roi_bin_grid_h = 3 and roi_bin_grid_w = 2 we will 
                # have overall 6 points we interpolate the values of and then average them to 
                # come up with a value for each of the 4 areas in pooled RoI region (which is
                # 2 x 2 if  pooled_height = pooled_width = 2)
                for iy in range(int(roi_bin_grid_h)):
                    # ph * bin_size_h - which square in RoI to pick vertically (on y axis)
                    # (iy + 0.5) * bin_size_h / roi_bin_grid_h - which of the roi_bin_grid_h points
                    # vertically to select within square 
                    yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                    for ix in range(int(roi_bin_grid_w)):
                        # pw * bin_size_w -  which square in RoI to pick horizontally (on x axis)
                        # (ix + 0.5) * bin_size_w / roi_bin_grid_w - which of the roi_bin_grid_w points
                        # vertically to select within square 
                        xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
                        for iz in range(int(roi_bin_grid_d)):
                            # pd * bin_size_d -  which square in RoI to pick horizontally (on x axis)
                            # (ix + 0.5) * bin_size_w / roi_bin_grid_w - which of the roi_bin_grid_w points
                            # vertically to select within square
                            zz = roi_start_d + pd * bin_size_d + (iz + 0.5) * bin_size_d / roi_bin_grid_d
                            
                            x = xx
                            y = yy
                            z = zz

                            x = tf.maximum(0,x)
                            y = tf.maximum(0,y)
                            z = tf.maximum(0,z)
                            x_low = tf.cast(x, 'int32')
                            y_low = tf.cast(y, 'int32')
                            z_low = tf.cast(z, 'int32')
                            
                            ## X
                            if (y_low >= height - 1):
                                y_high = y_low = height - 1
                                y = y_low
                            else:
                                y_high = y_low + 1
                            ## Y
                            if (x_low >= width-1):
                                x_high = x_low = width - 1
                                x = x_low
                            else:
                                x_high = x_low + 1

                            ## Z
                            if (z_low >= depth-1):
                                z_high = z_low = depth - 1
                                z = z_low
                            else:
                                z_high = z_low + 1
                            
                            ly = y - y_low
                            lx = x - x_low
                            lz = z - z_low

                            hy = 1 - ly
                            hx = 1 - lx
                            hz = 1 - lz

                            # w1 = hy * hx; w2 = hy * lx; w3 = ly * hx; w4 = ly * lx

                            pos1 = (y_low, x_low, z_low); pos2 = (y_low, x_high, z_low)
                            pos3 = (y_high, x_low, z_low); pos4 = (y_high, x_high, z_low)
                            pos5 = (y_low, x_low, z_high); pos6 = (y_low, x_high, z_high)
                            pos7 = (y_high, x_low, z_high); pos8 = (y_high, x_high, z_high)

                            w1 = hy*hx*hz
                            w2 = hy*lx*hz
                            w3 = ly*hx*hz
                            w4 = ly*lx*hz
                            w5 = hy*hx*lz
                            w6 = hy*lx*lz
                            w7 = ly*hx*lz
                            w8 = ly*lx*lz

                            precalc[pre_calc_index] = (pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8,
                                                        w1, w2, w3, w4, w5, w6, w7, w8)
            
                            pre_calc_index += 1
                        # print(x, y)
    return precalc



def crop_resize_one_image(image, box, crop_size):
    y1, x1, z1, y2, x2, z2 = box
    width = image.shape[0]
    height = image.shape[1]
    depth = image.shape[2]
    # denormalize
    roi_proposal = np.multiply(box, np.array([height, width, depth, height, width, depth]))
    roi_start_h, roi_start_w, roi_start_d, roi_end_h, roi_end_w, roi_end_d = roi_proposal

    # pooling regions size
    pooled_width = crop_size[0]
    pooled_height = crop_size[1]
    pooled_depth = crop_size[2]

    # RoI width, height and depth
    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    roi_depth = roi_end_d - roi_start_d

    # raw height and weight of each RoI sub-regions
    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width
    bin_size_d = roi_depth / pooled_depth

    # we divide each RoI sub-region into roi_bin_grid_h x roi_bin_grid_w x roi_bing_grid_d areas.
    # These will define the number of sampling points in each sub-region
    roi_bin_grid_h = np.ceil(roi_height / pooled_height)
    roi_bin_grid_w = np.ceil(roi_width / pooled_width)
    roi_bin_grid_d = np.ceil(roi_depth / pooled_depth)

    precalc = precalc_bilinear(height, width, depth, crop_size, roi_start_h, roi_start_w, roi_start_d,
                               bin_size_h, bin_size_w, bin_size_d, roi_bin_grid_h, roi_bin_grid_w, roi_bin_grid_d)

    count = max(roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d, 1)
    output = tf.zeros((pooled_height, pooled_width, pooled_depth))
    pre_calc_index = 0
    for ph in range(int(pooled_height)):
        for pw in range(int(pooled_width)):
            for pd in range(int(pooled_depth)):
                output_val = 0
                for iy in range(int(roi_bin_grid_h)):
                    for ix in range(int(roi_bin_grid_w)):
                        for iz in range(int(roi_bin_grid_d)):
                            (pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8,w1, w2, w3, w4, w5, w6, w7, w8) =  precalc[pre_calc_index] 
                            (y_low, x_low, z_low)   = pos1 
                            (y_low, x_high, z_low)  = pos2
                            (y_high, x_low, z_low)  = pos3
                            (y_high, x_high, z_low) = pos4
                            (y_low, x_low, z_high)  = pos5
                            (y_low, x_high, z_high) = pos6
                            (y_high, x_low, z_high) = pos7
                            (y_high, x_high, z_high)= pos8

                            output_val += w1 * image[y_low, x_low, z_low] + w2 * image[y_low, x_high, z_low] + \
                                            w3 * image[y_high, x_low, z_low] +  w4 * image[y_high, x_high, z_low] + \
                                            w5 * image[y_low, x_low, z_high] + w6 * image[y_low, x_high, z_high] + \
                                            w7 * image[y_high, x_low, z_high] +  w8 * image[y_high, x_high, z_high]

                            pre_calc_index += 1 

                # we do average pooling here
                output[ph, pw, pd] = tf.convert_to_tensor(output_val / count)
    return output 


def crop_and_resize_2nd(image, boxes, box_indices, crop_size):
    ## preprocess arguments
    xy_stack = tf.unstack(image, axis=3)
    depth = len(xy_stack)
    y1, x1, z1, y2, x2, z2 = tf.split(boxes, 6, axis=1)
    boxes_xy = tf.concat([x1, y1, x2, y2], axis=1)
    boxes_z = tf.concat([z1, z2], axis=1)

    cropped_xy = []
    crop_size_xy = (crop_size[0], crop_size[1])
    for stack in xy_stack:
        cropped_xy.append(tf.image.crop_and_resize(stack, boxes_xy, box_indices, crop_size))        


def crop_and_resize(image, boxes, box_indices, crop_size):
   pass