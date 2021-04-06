import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from yolov4.tool.torch_utils import *
from yolov4.tool.yolo_layer import YoloLayer
from yolov4.models import Conv_Bn_Activation

## Route Class
## From https://github.com/jinyeom/smol/blob/f24cedfce0f2aeafb2a589c9da52e4a913b431c1/smol/modules.py#L57

class Route(nn.Module):
    def __init__(self, groups: int = 1, group_id: int = 0):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def forward(self, *tensors: Tensor) -> Tensor:
        output = []
        for tensor in tensors:
            chunks = torch.chunk(tensor, self.groups, dim=1)
            output.append(chunks[self.group_id])
        return torch.cat(output, dim=1)

## TinyYolo Pytorch Model

## 
class TinyYolo(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        ## Vars
        self.inference = inference
        self.num_classes = 1
        self.n_output_ch = (4 + 1 + self.num_classes) * 3

        ## Layers
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 2, 'leaky')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'leaky')
        self.conv3 = Conv_Bn_Activation(64, 64, 3, 1, 'leaky')
        self.route1 = Route(groups=2, group_id=1)
        self.conv4 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(32, 32, 3, 1, 'leaky')
        self.route2 = Route()
        self.conv6 = Conv_Bn_Activation(64, 64, 1, 1, 'leaky')
        self.route3 = Route()
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv7 = Conv_Bn_Activation(128, 128, 3, 1, 'leaky')
        self.route4 = Route(groups=2, group_id=1)
        self.conv8 = Conv_Bn_Activation(64, 64, 3, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(64, 64, 3, 1, 'leaky')
        self.route5 = Route()
        self.conv10 = Conv_Bn_Activation(128, 128, 1, 1, 'leaky')
        self.route6 = Route()
        self.max2 = nn.MaxPool2d(2, 2)
        self.conv11 = Conv_Bn_Activation(256, 256, 3, 1, 'leaky')
        self.route7 = Route(groups=2, group_id=1)
        self.conv12 = Conv_Bn_Activation(128, 128, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(128, 128, 3, 1, 'leaky')
        self.route8 = Route()
        self.conv14 = Conv_Bn_Activation(256, 256, 1, 1, 'leaky')
        self.route9 = Route()
        self.max3 = nn.MaxPool2d(2, 2)
        self.conv15 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')
        #######
        self.conv16 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')

        self.conv18 = Conv_Bn_Activation(512, self.n_output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=self.num_classes,
                                anchors=[10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319],
                                num_anchors=6, stride=8)

        self.route10 = Route()
        self.conv19 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.route11 = Route()
        self.conv20 = Conv_Bn_Activation(384, 256, 3, 1, 'leaky')
        self.conv21 = Conv_Bn_Activation(256, self.n_output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[1, 2, 3], num_classes=self.num_classes,
                                anchors=[10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319],
                                num_anchors=6, stride=16)

    def forward(self, input):
        x0 = self.conv1(input)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1)

        x3 = self.route1(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        x6 = self.route2(x5, x4)
        x7 = self.conv6(x6)

        x8 = self.route3(x2, x7)
        x9 = self.max1(x8)
        x10 = self.conv7(x9)

        x11 = self.route4(x10)
        x12 = self.conv8(x11)
        x13 = self.conv9(x12)

        x14 = self.route5(x13, x12)
        x15 = self.conv10(x14)

        x16 = self.route6(x10, x15)
        x17 = self.max2(x16)
        x18 = self.conv11(x17)

        x19 = self.route7(x18)
        x20 = self.conv12(x19)
        x21 = self.conv13(x20)

        x22 = self.route8(x21, x20)
        x23 = self.conv14(x22)

        x24 = self.route9(x18, x23)
        x25 = self.max3(x24)
        x26 = self.conv15(x25)

        ##################################

        x27 = self.conv16(x26)
        x28 = self.conv17(x27)
        x29 = self.conv18(x28)

        x31 = self.route10(x27)
        x32 = self.conv19(x31)
        x33 = self.upsample1(x32)

        x34 = self.route11(x33, x23)
        x35 = self.conv20(x34)
        x36 = self.conv21(x35)

        if self.inference:
            yolo1 = self.yolo1(x29)
            yolo2 = self.yolo2(x36)

            return get_region_boxes([yolo1, yolo2])
        else:
            return [x29, x36]
