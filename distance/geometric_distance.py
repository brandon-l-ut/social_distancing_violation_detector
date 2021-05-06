import math

class Camera_Geom:
    def __init__(self, cfg):
        self.w_focal = cfg.w_focal ## mm
        self.h_focal = cfg.h_focal ## mm
        self.w_sens = cfg.w_sens ## mm
        self.h_sens = cfg.h_sens ## mm
        self.w_img = cfg.w_img 
        self.h_img = cfg.h_img 
        self.social_distance = cfg.social_distance #mm

        self.pixel_size = .5 * ((self.w_sens / self.w_img) + (self.h_sens / self.h_img))
        ## Average human height
        self.avg_height = 1725 #mm

    ## assumes bboxes are x1y1, x2y2
    def geometric_distance(self, bbox):
        # returns estimated distance in mm
        ydif = bbox[3] - bbox[1]
        pixel_height = abs(ydif)
        distance = self.avg_height * self.h_focal / (pixel_height * self.pixel_size)
        
        return distance

    def compute_pair_violation(self, bbox1, bbox2):
        ## Determine whether 2 bboxes are violating social distancing
        d1 = self.geometric_distance(bbox1)
        d2 = self.geometric_distance(bbox2)

        p1_x_center = abs(bbox1[0] + bbox1[2]) / 2
        p2_x_center = abs(bbox2[0] + bbox2[2]) / 2

        w = self.w_sens * abs(p1_x_center - p2_x_center) * self.pixel_size / self.w_focal
        social_distance = math.sqrt(w**2 + (d1 - d2)**2)
        print("Social Distance: ", social_distance)
        if social_distance < self.social_distance:
            return True
        else:
            return False

    def compute_violations(self, bboxes):
        ## modifies list of bboxes
        ## add 1 to end of list indicating violated
        ## add 0 to end of list indicating not violated
        [bboxes[i].append(0) for i in range(len(bboxes))]

        for p1 in range(len(bboxes)):
            if (bboxes[p1][4] == 1):
                continue
            for p2 in range(p1+1, len(bboxes)):
                if self.compute_pair_violation(bboxes[p1], bboxes[p2]):
                    bboxes[p1][4] = 1
                    bboxes[p2][4] = 1
