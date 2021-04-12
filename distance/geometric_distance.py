import math

class Camera_Geom:
    def __init__(self, w_focal, h_focal, w_sens, h_sens, w_img, h_img):
        self.w_focal = w_focal ## mm
        self.h_focal = h_focal ## mm
        self.w_sens = w_sens ## mm
        self.h_sens = h_sens ## mm
        self.w_img = w_img 
        self.h_img = h_img 

        self.pixel_size = .5 * ((self.w_sens / self.w_img) + (self.h_sens / self.h_img))

        self.avg_height = 1600 #mm
        self.social_distance = 1800 #mm

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

        w = self.w_sens * abs(bbox1[0] - bbox2[0]) * self.pixel_size / self.w_focal
        social_distance = math.sqrt(w**2 + (d1 - d2)**2)

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
            for p2 in range(p1, len(bboxes)):
                if self.compute_pair_violation(bboxes[p1], bboxes[p2]):
                    bboxes[p1][4] = 1
                    bboxes[p2][4] = 1
