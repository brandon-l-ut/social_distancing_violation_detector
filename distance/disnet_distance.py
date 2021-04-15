import math
import numpy as np

class Camera_Disnet:
    def __init__(self, cfg):
        self.social_distance = cfg.social_distance
        self.h_img = cfg.h_img
        if cfg.inverted:
            self.k_matrix = np.array(cfg.k_matrix)
        else:
            tmp_matrix = np.array(cfg.k_matrix)
            self.k_matrix = np.linalg.inv(tmp_matrix)
        self.dist = 2743

    def get_distance(self, im_coords):
        return self.dist

    ## Called by sd_detector
    def compute_violations(self, bboxes):
        ## modifies list of bboxes
        ## add 1 to end of list indicating violated
        ## add 0 to end of list indicating not violated
        [bboxes[i].append(0) for i in range(len(bboxes))]
        
        # Get real world coords of each person
        r_coords = []
        for bbox in bboxes:
            im_coords = np.array([bbox[2], self.h_img - bbox[3], 1])
            print("im coords", im_coords)
            distance = self.get_distance(im_coords)
            norm_coord = np.matmul(self.k_matrix, im_coords)
            print("norm coord", norm_coord)
            r_coord = norm_coord * distance / np.linalg.norm(norm_coord)
            print("r coord:", r_coord)
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
