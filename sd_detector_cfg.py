## Configuration File for Social Distance detector
from enum import Enum

import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

## Location of object detection configuration file
Cfg.cfg_file = os.path.join(_BASE_DIR, 'cfg', 'regular_prune_0.7_yolov4-tiny-person.cfg')
## Location of object detection weights
Cfg.weight_file = os.path.join(_BASE_DIR, 'weights', 'best_regular_prune_0.7_finetune.pt')
## Pretrained model input width/height
Cfg.model_h = 416
Cfg.model_w = 416
## Use Cuda GPU
Cfg.cuda = False

## Configurable social distance. Default set to 1800 mm ~ 6 ft
Cfg.social_distance = 1800

## Video or image file to detect social distancing in
#Cfg.file_path = "input_images/vid_short.mp4"
Cfg.file_path = "input_images/test_0.jpg"

## File type of media 
Cfg.video = False
## Whether to print FPS to console when running a video
Cfg.show_fps = False

## Widgh / height of image / video
Cfg.h_img = 720
Cfg.w_img = 540

## Enable tracking people, displays index of person in image
Cfg.tracking = True
## Ability to save output of social distancing run
Cfg.save_output = True
## Location and name social distancing results should be saved to 
Cfg.output_path = "output.jpg"

##################################################################################

class Distance_Methods(Enum):
    Geometric = 1
    IPM = 2
    Disnet = 3

## Method of estimating social distance between people. Must be from the above enum
## Our method is coded as 'Disnet' in the Distance_Methods Enum
Cfg.distance_calculation = Distance_Methods.Disnet

## Necessary parameters for estimating social distance using CV geometry technique
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
## whether the provided transformation matrix is already inverted. 
Cfg.inverted = False
# Intrinsic matrix-K
# Rotation matrix-R
# Translation matrix-T
# Need to put K*R*T below. cut out 3rd column

## Matrix for apartment tests
#Cfg.calib_matrix = [[ 5.80616134e+02,  1.34470135e+02,   -1.01781279e+06] ,
# [ 0.00000000e+00,  6.82233091e+02,  -1.12244972e+05] ,
# [ 0.00000000e+00,  5.00000000e-01,  -3.78453101e+03]]

## Matrix for park tests
Cfg.calib_matrix = [[ 5.80616134e+02,  1.13659069e+02,   -7.58101505e+05],
 [ 0.00000000e+00,  6.77398362e+02,   -2.64521288e+05],
 [ 0.00000000e+00 , 4.22618262e-01,   -2.81884712e+03]]

## Necessary parameters for Disnet social distancing detection. 
## need camera intrinsic Matrix K
Cfg.k_matrix = [[580.6161339,    0.,         268.94026972],
 [  0.,         577.98851373, 363.36070951],
 [  0. ,          0.,           1.        ]]

## Location of disnet weights to use when estimating social distance
Cfg.disnet_weights = "distance/disnet/Disnet-best.pth"