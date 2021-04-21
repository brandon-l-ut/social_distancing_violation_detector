## Configuration File for Social Distance detector
from enum import Enum

import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

## Location of object detection configuration file
Cfg.cfg_file = os.path.join(_BASE_DIR, 'yolov4', 'cfg', 'yolov4-tiny.cfg')
## Location of object detection weights
Cfg.weight_file = os.path.join(_BASE_DIR, 'weights', 'yolov4-tiny.weights')
## Pretrained model input width/height
Cfg.model_h = 416
Cfg.model_w = 416
## Use Cuda GPU
Cfg.cuda = False

## Configurable social distance. Default set to 1800 mm ~ 6 ft
Cfg.social_distance = 1800

## Video or image file to detect social distancing in
#Cfg.file_path = "data/train2014/COCO_train2014_000000023548"
Cfg.file_path = "test_img/vid_short.mp4"
#Cfg.file_path = "test_img/geometric/resized/test_2.jpg"
## File type of media 
Cfg.video = True

## Dimensions of image / video
Cfg.h_img = 720
Cfg.w_img = 1280

## Enable tracking people, displays index of person in image
Cfg.tracking = True
## Ability to save output of social distancing run
Cfg.save_output = False
## Location social distancing results should be saved to 
Cfg.output_path = "out.jpg"

class Distance_Methods(Enum):
    Geometric = 1
    IPM = 2
    Disnet = 3

## Method of estimating social distance between people. Must be from the above enum
Cfg.distance_calculation = Distance_Methods.Disnet

## Necessary parameters for estimating social distance using CV geometry 
## Default parameters are for an iPhone 8
## Focal width in mm
Cfg.w_focal = 3.99
## Focal height in mm
Cfg.h_focal = 3.99
## Width of camera sensor in mm
Cfg.w_sens = 4.80
## Height of camera sensor in mm
Cfg.h_sens = 3.600

## Necessary parameters for performing inverse perspective mapping
## 
Cfg.inverted = False
# Intrinsic matrix-K
# Rotation matrix-R
# Translation matrix-T
# Need to put K*R*T below. cut out 3rd column
Cfg.calib_matrix = [[ 5.80616134e+02,  1.34470135e+02,   -1.01781279e+06] ,
 [ 0.00000000e+00,  6.82233091e+02,  -1.12244972e+05] ,
 [ 0.00000000e+00,  5.00000000e-01,  -3.78453101e+03]]

  #Cfg.intrinsic_matrix = [[ 5.06338752e+02,  1.37705204e+02,  2.38512409e+02, -1.04229923e+06],
 #[ 0.00000000e+00,  6.12550496e+02,  5.19641209e+01, -2.27083208e+05],
 #[ 0.00000000e+00,  5.00000000e-01,  8.66025404e-01, -3.78453101e+03]]

 ## For Disnet
Cfg.k_matrix = [[580.6161339,    0.,         268.94026972],
 [  0.,         577.98851373, 363.36070951],
 [  0. ,          0.,           1.        ]]

Cfg.disnet_weights = "distance/disnet/Disnet.pth"