from mrcnn.utils import nms_3d
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------------
# Scratch implementation
bboxes = np.array([(165, 127, 1, 296, 455, 1), (148, 142, 1, 257, 459, 1), (142, 137, 1, 270, 465, 1),
                   (129, 122, 1, 302, 471, 1), (327, 262, 1, 604, 465, 1), (349, 253, 1, 618, 456, 1),
                   (369, 248, 1, 601, 470, 1)])
pscores = np.array([0.8,0.95,0.81,0.85,0.94,0.83,0.82])
print(nms_3d(bboxes,pscores,0.3, 5))



bboxes = np.array([(165,127,296,455),(148,142,257,459),(142,137,270,465),(129,122,302,471),
                   (327,262,604,465),(349,253,618,456),(369,248,601,470)])


# ----------------------------------------------------------------------------------
# Tensorflow implementation
import tensorflow as tf

def nms(boxes, scores, threshold):
            indices = tf.image.non_max_suppression(
                boxes, scores, 5,
                threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(5 - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

print(nms(bboxes,pscores,0.3))