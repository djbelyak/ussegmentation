import torch.nn as nn
import torch.nn.functional as F


class EmptyNet(nn.Module):
    name = "empty"

    def __init__(self):
        super(EmptyNet, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x
