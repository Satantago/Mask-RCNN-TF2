import open3d.ml.torch as ml3d
import numpy as np

boxes = np.array([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                  [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                  [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                  [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                  [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                  [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                 dtype=np.float32)
scores = np.array([3, 1.1, 5, 2, 1, 0], dtype=np.float32)
nms_overlap_thresh = 0.7
keep_indices = ml3d.ops.nms(boxes, scores, nms_overlap_thresh)
print(keep_indices)