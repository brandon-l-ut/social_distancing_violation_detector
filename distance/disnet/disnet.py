import math

import torch
from torch import nn
import torch.nn.functional as F

class Disnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(6, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, 50)
        self.lin4 = nn.Linear(50, 50)
        self.lin5 = nn.Linear(50, 1)

    def preprocess_bbox(self, bbox_norm, inference):
        bbox_norm = bbox_norm.tolist()
        inp = []

        if inference:
            bbox = bbox_norm
            # width height and diagonal of bbox
            w = 1 / (abs(bbox[0] - bbox[2]))
            h = 1 / (abs(bbox[1] - bbox[3]))
            d = 1 / (math.sqrt((bbox[0] - bbox[2])**2 + (abs(bbox[1] - bbox[3])**2)))

            # width, height, breadth of person in mm
            p_width = .570
            p_height = 1.800
            p_breadth = .300

            inp.append([w, h, d, p_width, p_height, p_breadth])
        else:
            for bbox in bbox_norm:
                # width height and diagonal of bbox
                w = 1 / (abs(bbox[0] - bbox[2]))
                h = 1 / (abs(bbox[1] - bbox[3]))
                d = 1 / (math.sqrt((bbox[0] - bbox[2])**2 + (abs(bbox[1] - bbox[3])**2)))

                # width, height, breadth of person in mm
                p_width = .550
                p_height = 1.750
                p_breadth = .300

                inp.append([w, h, d, p_width, p_height, p_breadth])
        return torch.tensor(inp)

    def forward(self, bbox_norm, inference):

        x0 = self.preprocess_bbox(bbox_norm, inference)
        
        x1 = self.lin1(x0)
        x2 = self.lin2(x1)
        x3 = self.lin3(x2)
        x4 = self.lin4(x3)
        x5 = self.lin5(x4)
        # gives distance in meters
        return x5
