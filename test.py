## try to setup yolo inference


import torch
from yolov4.demo import detect_cv2
from TinyYoloModel import TinyYolo

##detect_cv2("./yolov4/cfg/yolov4.cfg", "weights/yolov4.weights", "data/val2014/COCO_val2014_000000262148.jpg")

if __name__ == "__main__":
    import sys
    import cv2
    
    weightfile = "tinyyolo_10.pth"
    imgfile = "data/train2014/COCO_train2014_000000000110.jpg"
    model = TinyYolo(inference=True)
    width = 416
    height = 416

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    
    namesfile = 'yolov4/data/coco.names'
    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)