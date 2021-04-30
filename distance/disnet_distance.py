import math
import numpy as np

import torch

from distance.disnet.disnet import Disnet

class Camera_Disnet:
    def __init__(self, cfg):
        self.social_distance = cfg.social_distance
        self.h_img = cfg.h_img

        self.model = Disnet()
        self.model.load_state_dict(torch.load(cfg.disnet_weights))

        if cfg.inverted:
            self.k_matrix = np.array(cfg.k_matrix)
        else:
            tmp_matrix = np.array(cfg.k_matrix)
            self.k_matrix = np.linalg.inv(tmp_matrix)
        #self.dist = 2743

    def get_distance(self, im_coords):
        with torch.no_grad():
            ## Disnet returns distance in meters - converting to mm
            return self.model(im_coords, True) * 1000

    ## Called by sd_detector
    def compute_violations(self, bboxes):
        ## modifies list of bboxes
        ## add 1 to end of list indicating violated
        ## add 0 to end of list indicating not violated
        [bboxes[i].append(0) for i in range(len(bboxes))]
        
        # Get real world coords of each person
        r_coords = []
        for bbox in bboxes:
            im_coord1 = bbox[:2] + [1]
            im_coord2 = bbox[2:4] + [1]

            norm_coord1 = np.matmul(self.k_matrix, im_coord1).tolist()
            norm_coord2 = np.matmul(self.k_matrix, im_coord2).tolist()

            bbox_torch = torch.tensor(norm_coord1[:2] + norm_coord2[:2])
            distance = self.get_distance(bbox_torch).tolist()[0][0]

            x_centered_coord = [(norm_coord1[0] + norm_coord2[0]) / 2, norm_coord2[2], 1.0]
            
            r_coord = np.array(x_centered_coord) * distance / np.linalg.norm(np.array(x_centered_coord))
            #print("r coord:", r_coord)
            r_coords.append(list(r_coord))
            
        for p1 in range(len(bboxes)):
            if (bboxes[p1][4] == 1):
                continue
            for p2 in range(p1+1, len(bboxes)):
                xdif = r_coords[p1][0] - r_coords[p2][0]
                zdif = r_coords[p1][2] - r_coords[p2][2]
                social_dist = math.sqrt(xdif**2 + zdif**2)
                print("Social distance:", social_dist)

                if social_dist < self.social_distance:
                    bboxes[p1][4] = 1
                    bboxes[p2][4] = 1
