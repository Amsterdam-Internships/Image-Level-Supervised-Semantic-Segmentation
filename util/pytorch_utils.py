import torch.nn as nn


class Interpolate(nn.Module):
    """
    Since nn.Interpolate does not exist in PyTorch, but nn.Upsample is deprecated.
    Removes warning.
    """
    def __init__(self, size, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x
