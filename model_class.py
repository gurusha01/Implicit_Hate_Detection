import torch.nn as nn
import torch

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()

    def forward(self, image, text):
        return [0]*13