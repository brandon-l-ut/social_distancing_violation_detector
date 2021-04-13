##python yolov4/evaluate_on_coco.py -f weights/yolov4.pth -g 1 
# -dir data/val2014 -gta data/annotations/instances_val2014_person.json 
# -c yolov4/cfg/yolov4.cfg -w weights/yolov4.weights


import math
import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2

from yolov4.tool.darknet2pytorch import Darknet
from yolov4.tool.utils import *
from yolov4.tool.torch_utils import *

def convert_cat_id_and_reorientate_bbox(single_annotation):
    cat = single_annotation['category_id']
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    single_annotation['category_id'] = cat
    return single_annotation

def test(model, use_cuda, annotations_file_path):
    with open(annotations_file_path) as annotations_file:
            annotations = json.load(annotations_file)

    images = annotations["images"]
    resFile = 'data/coco_val_outputs.json'

    boxes_json = []
    time_tot = float(0)

    for i, image_annotation in enumerate(images):
        logging.info("currently on image: {}/{}".format(i + 1, len(images)))
        image_file_name = image_annotation["file_name"]
        image_id = image_annotation["id"]
        image_height = image_annotation["height"]
        image_width = image_annotation["width"]

        # open and resize each image first
        img = cv2.imread("data/val2014/" + image_file_name)
        
        sized = cv2.resize(img, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(model, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        time_tot += finish - start
        if type(boxes[0]) == list:
            for box in boxes[0]:
                box_json = {}
                category_id = box[-1]
                if category_id != 0:
                    continue

                score = box[-2]
                bbox_normalized = box[:4]
                box_json["image_id"] = int(image_id)
                box_json["category_id"] = int(category_id)
                bbox = []
                for i, bbox_coord in enumerate(bbox_normalized):
                    modified_bbox_coord = float(bbox_coord)
                    if i % 2:
                        modified_bbox_coord *= image_height
                    else:
                        modified_bbox_coord *= image_width
                    modified_bbox_coord = round(modified_bbox_coord, 2)
                    bbox.append(modified_bbox_coord)
                tmp = [bbox[0], bbox[1], round(bbox[2] - bbox[0], 4), round(bbox[3] - bbox[1], 4)]
                box_json["bbox"] = tmp
                box_json["score"] = round(float(score), 3)
                boxes_json.append(box_json)

    boxes_json.sort(key=lambda x: x["image_id"])
    boxes_json = list(map(convert_cat_id_and_reorientate_bbox, boxes_json))
    with open(resFile, 'w') as outfile:
        json.dump(boxes_json, outfile)

    cocoGt = COCO(annotations_file_path)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    logging.info("Total time: {}".format(time_tot))

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    annotations_file_path = "data/annotations/instances_val2014_person.json"
    weightfile = "weights/yolov4.weights"
    cfgfile = "yolov4/cfg/yolov4.cfg"
    use_cuda = True
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Using device {device}')

    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)

    
    if use_cuda:
        m.cuda()
    
    test(m, use_cuda, annotations_file_path)
