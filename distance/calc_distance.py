
class camera:
    def __init__(self, x_focal_l, y_focal_l):
        self.x_focal = x_focal_l
        self.y_focal = y_focal_l
        self.avg_height = 1.6 #meters
        self.social_distance = 1.8 #meters

    ## assumes bboxes are x1y1, x2y2
    def geometric_distance(self, bbox):
        ydif = bbox[3] - bbox[1]
        pixel_height = max(-1 * ydif, ydif)
        distance = self.avg_height * self.y_focal / pixel_height

        return distance

    def compute_pair_violation(self, bbox1, bbox2):
        ## Determine whether 2 bboxes are violating social distancing

    def compute_violations(self, bboxes):
        ## return list of bboxes
        ## add 1 to end of list indicating violated
        ## add 0 to end of list indicating not violated
        [bboxes[i].append(0) for i in range(len(bboxes))]

        for p1 in range(len(bboxes)):
            if (bboxes[p1] == 1):
                continue
            for p2 in range(1, len(bboxes)):
                if compute_pair_violation(bboxes[p1], bboxes[p2]):
                    bboxes[p1] = 1
                    bboxes[p2] = 1
