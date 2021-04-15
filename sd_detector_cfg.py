## Configuration File for Social Distance detector
from enum import Enum

class Distance_Methods(Enum):
    Geometric = 1
    IPM = 2

import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.cfg_file = os.path.join(_BASE_DIR, 'yolov4', 'cfg', 'yolov4.cfg')
Cfg.weight_file = os.path.join(_BASE_DIR, 'weights', 'yolov4.weights')
Cfg.cuda = True

## 1800 mm ~ 6 ft
Cfg.social_distance = 1800 #mm

# Vido or image file to detect social distancing in
#Cfg.file_path = "data/train2014/COCO_train2014_000000023548"
#Cfg.file_path = "vid_short.mp4"
Cfg.file_path = "test_img/ipm/resized/test_2.jpg"
Cfg.h_img = 720
Cfg.w_img = 540
Cfg.video = False
Cfg.save_output = True
Cfg.output_path = "out.jpg"

Cfg.distance_calculation = Distance_Methods.IPM

## For Geometric - iphone 8
Cfg.w_focal = 3.99
Cfg.h_focal = 3.99
Cfg.w_sens = 4.80
Cfg.h_sens = 3.600

## For IPM
Cfg.inverted = False
# Intrinsic matrix-K
# Rotation matrix-R
# Translation matrix-T
# Need to put K*R*T below
Cfg.intrinsic_matrix = [[ 5.06338752e+02,  1.37705204e+02, -1.04229923e+06],
 [ 0.00000000e+00,  6.12550496e+02,  -2.27083208e+05],
 [ 0.00000000e+00,  5.00000000e-01,  -3.78453101e+03]]

  #Cfg.intrinsic_matrix = [[ 5.06338752e+02,  1.37705204e+02,  2.38512409e+02, -1.04229923e+06],
 #[ 0.00000000e+00,  6.12550496e+02,  5.19641209e+01, -2.27083208e+05],
 #[ 0.00000000e+00,  5.00000000e-01,  8.66025404e-01, -3.78453101e+03]]