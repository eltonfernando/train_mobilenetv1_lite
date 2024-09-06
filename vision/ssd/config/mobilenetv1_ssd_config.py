import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 224
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(14, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(7, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(4, 56, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(2, 110, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(1, 224, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 244, SSDBoxSizes(285, 330), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)