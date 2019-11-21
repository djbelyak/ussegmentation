import torch.nn as nn


class EmptyNet(nn.Module):
    name = "empty"

    def __init__(self, num_classes=3):
        super(EmptyNet, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(
            3, num_classes, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        input_size = x.size()
        x = self.transposed_conv(x, output_size=input_size)
        return x
