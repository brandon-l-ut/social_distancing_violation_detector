import cv2
import torch
import numpy as np
import time

from torch.autograd import Variable

from distance.geometric_distance import Camera_Geom

from yolov4.tool import utils
from yolov4.tool.darknet2pytorch import Darknet

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    
    t1 = time.time()

    output = model(img)

    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    return utils.post_processing(img, conf_thresh, nms_thresh, output)

def post_process_bboxes(bboxes, w, h):
    ## convert bboxes back to pixels, filter out non-people
    post_bboxes = []
    for bbox in bboxes:
        if bbox[6] != 0:
            continue
        x1 = bbox[0] * w
        y1 = bbox[1] * h
        x2 = bbox[2] * w
        y2 = bbox[3] * h
        post_bboxes.append([x1, y1, x2, y2])

    return post_bboxes
        
def draw_boxes(img, bboxes):
    img = np.copy(img)
    red = (0, 0, 255)
    green = (0, 255, 0)

    for bbox in bboxes:
        if bbox[4] == 1:
            color = red
        else:
            color = green

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color)
    
    cv2.imwrite("tmp.jpg", img)

def sd_frame(img_fname):
    m = Darknet("yolov4/cfg/yolov4.cfg")
    m.load_weights("weights/yolov4.weights")

    use_cuda = True
    if use_cuda:
        m.cuda()

    img = cv2.imread(img_fname)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    c = Camera_Geom(4.15, 4.15, 4.80, 3.60, img.shape[1], img.shape[0])

    bboxes = do_detect(m, sized, 0.4, 0.6, use_cuda)[0]
    post_bboxes = post_process_bboxes(bboxes, img.shape[1], img.shape[0])

    c.compute_violations(post_bboxes)
    draw_boxes(img, post_bboxes)
    


if __name__ == '__main__':
    sd_frame("data/train2014/COCO_train2014_000000000110.jpg")