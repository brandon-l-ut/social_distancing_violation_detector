
class camera:
    def __init__(self, x_focal_l, y_focal_l):
        self.x_focal = x_focal_l
        self.y_focal = y_focal_l
        self.avg_height = 5

    ## assumes bboxes are x1y1, x2y2
    def geometric_distance(self, bbox):
        ydif = bbox[3] - bbox[1]
        pixel_height = max(-1 * ydif, ydif)
        distance = self.avg_height * self.y_focal / pixel_height

        return distance