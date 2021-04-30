import cv2
import torch
import numpy as np
import time

from torch.autograd import Variable

from pruning.models import Darknet as Dn_pruned
from pruning.utils.utils import output_to_target, non_max_suppression

from yolov4.tool.darknet2pytorch import Darknet as Dn_oob
from yolov4.tool import utils

from distance.geometric_distance import Camera_Geom
from distance.ipm_distance import Camera_IPM
from distance.disnet_distance import Camera_Disnet
from distance.centroid_tracking import Centroid_Tracking

from sd_detector_cfg import Distance_Methods

class SD_Detector():
    def __init__(self, cfg):
        self.file_fname = cfg.file_path
        self.video = cfg.video
        self.w_img = cfg.w_img
        self.h_img = cfg.h_img
        self.cuda = cfg.cuda
        self.save_output = cfg.save_output
        self.output_path = cfg.output_path
        self.model_w = cfg.model_w
        self.model_h = cfg.model_h
        self.tracking = cfg.tracking
        self.fps_vals = []
        self.show_fps = cfg.show_fps

        if cfg.weight_file.endswith('.pt'):
            self.model = Dn_pruned(cfg.cfg_file)
            weights = torch.load(cfg.weight_file)
            weights['model'] = {k: v for k, v in weights['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
            self.model.load_state_dict(weights['model'], strict=False)
            self.pt_weights = True
        else: 
            self.model = Dn_oob(cfg.cfg_file)
            self.model.load_weights(cfg.weight_file)
            self.pt_weights = False

        if cfg.cuda:
            self.model.cuda()

        if self.tracking:
            self.tracker = Centroid_Tracking()

        if cfg.distance_calculation is Distance_Methods.Geometric:
            self.distance_calculator = Camera_Geom(cfg)
        elif cfg.distance_calculation is Distance_Methods.IPM:
            self.distance_calculator = Camera_IPM(cfg)
        elif cfg.distance_calculation is Distance_Methods.Disnet:
            self.distance_calculator = Camera_Disnet(cfg)  
        else:
            print("Error")
            exit()


    def do_detect(self, model, img, conf_thresh, nms_thresh, use_cuda=1):
        model.eval()

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

        if self.pt_weights:
            inf_out, _, _ = model(img)
            output = non_max_suppression(inf_out, conf_thres=conf_thresh, iou_thres=nms_thresh, multi_label=True)
        else:
            inf_out = model(img)
            output = utils.post_processing(img, conf_thresh, nms_thresh, inf_out)

        return output


    def post_process_bboxes(self,bboxes):
        ## convert bboxes back to pixels, filter out non-people
        post_bboxes = []
        if not self.pt_weights:
            for bbox in bboxes:
                if bbox[6] != 0:
                    continue
                x1 = bbox[0] * self.w_img
                y1 = bbox[1] * self.h_img
                x2 = bbox[2] * self.w_img
                y2 = bbox[3] * self.h_img
                post_bboxes.append([x1, y1, x2, y2])
        else:
            x_scale = self.w_img / self.model_w
            y_scale = self.h_img / self.model_h
            for bbox in bboxes:
                x1 = bbox[0] * x_scale
                y1 = bbox[1] * y_scale
                x2 = bbox[2] * x_scale
                y2 = bbox[3] * y_scale
                post_bboxes.append([x1, y1, x2, y2])
        return post_bboxes
            
    def draw_boxes(self, img, bboxes):
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
            if self.tracking:
                img = cv2.putText(img, "Person {}".format(bbox[5]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 1)
        
        return img

    def sd_frame(self, img):
        # assumes cv2 img
        if self.show_fps:
            t1 = time.time()
        sized = cv2.resize(img, (self.model_w, self.model_h))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        bboxes = self.do_detect(self.model, sized, 0.4, 0.6, self.cuda)[0]
        if bboxes is None or len(bboxes) == 0:
            return (img)
        if not self.pt_weights:
            post_bboxes = self.post_process_bboxes(bboxes)
        else:
            post_bboxes = self.post_process_bboxes(bboxes[:,:4].detach().cpu().numpy().tolist())

        self.distance_calculator.compute_violations(post_bboxes)

        if self.tracking:
            post_bboxes = self.tracker.tracking_frame(post_bboxes)

        ret_img = self.draw_boxes(img, post_bboxes)

        if self.show_fps:
            t2 = time.time()
            if t1 != t2:
                print("FPS: {}.".format(1 / (t2 - t1)))
                self.fps_vals.append(1 / (t2 - t1))
        return ret_img

    def sd_video(self):
        cap = cv2.VideoCapture(self.file_fname)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        i_frame = 0
        if (self.save_output):
            save_file = cv2.VideoWriter(self.output_path, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (self.w_img, self.h_img))

        while (cap.isOpened()):
            i_frame += 1
            ret, frame = cap.read()

            ## Need to change this based on how fast object detection is
            if i_frame % 2 == 1:
                continue
            
            if ret:
                img = self.sd_frame(frame)
                cv2.imshow("Social Distancing Results", img)
                if self.save_output:
                    save_file.write(img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break            

        if self.save_output:
            save_file.release()

        cap.release()
        cv2.destroyAllWindows()

        if self.show_fps:
            print("Average FPS: {}.".format(np.average(self.fps_vals)))

    def detect_social_distance(self):
        if self.video:
            self.sd_video()
        else:
            img = cv2.imread(self.file_fname)
            img = self.sd_frame(img)
            cv2.imshow("Social Distancing Results", img)

            if self.save_output:
                cv2.imwrite(self.output_path, img)

            cv2.waitKey(0)
            #closing all open windows 
            cv2.destroyAllWindows() 

from sd_detector_cfg import Cfg

if __name__ == '__main__':

    detector = SD_Detector(Cfg)
    detector.detect_social_distance()
