## try to setup yolo inference


import torch
from yolov4.demo import detect_cv2

detect_cv2("./yolov4/cfg/yolov4.cfg", "weights/yolov4.weights", "data/val2014/COCO_val2014_000000262148.jpg")