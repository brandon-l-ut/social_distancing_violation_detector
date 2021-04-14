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
Cfg.file_path = "vid_short.mp4"
Cfg.h_img = 720
Cfg.w_img = 1280
Cfg.video = True
Cfg.save_output = True
Cfg.output_path = "out.avi"

Cfg.distance_calculation = Distance_Methods.Geometric

## For Geometric 
Cfg.w_focal = 4.15
Cfg.h_focal = 4.15
Cfg.w_sens = 4.80
Cfg.h_sens = 3600

## For IPM
Cfg.inverted = True
# Intrinsic matrix-K
# Rotation matrix-R
# Translation matrix-T
# Need to put K*R*T below, without 4th column
Cfg.intrinsic_matrix = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]