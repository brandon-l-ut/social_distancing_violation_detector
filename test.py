## try to setup yolo inference


import torch
from yolov4.demo import detect_cv2

detect_cv2("./yolov4/cfg/yolov4.cfg", "yolov4.weights", "./yolov4/data/dog.jpg")