import torch
from torch import nn
import torch.nn.functional as F

class Disnet(nn.module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(6, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 100)
        self.lin4 = nn.Linear(100, 100)
        self.lin5 = nn.Linear(100, 1)

    def forward(self, input):
        x1 = self.lin1(input)
        x2 = self.lin2(x1)
        x3 = self.lin3(x2)
        x4 = self.lin4(x3)
        x5 = self.lin5(x4)

        return x5
