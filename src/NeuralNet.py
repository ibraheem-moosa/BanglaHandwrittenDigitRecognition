import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import leaky_relu
from torch.nn.functional import dropout
import torchvision


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # (num of channels, num of filters, filter size)
                                            # Image size reduces by 4 on each dim after applying filter of size 5x5
        self.pool = nn.MaxPool2d(2, 2)      # Image size after pooling scales down by 2 on each dim
        self.conv2 = nn.Conv2d(6, 16, 5)    # num of channels is equal to num of filters of previous conv layer
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))  # bs x 6 x23x23
        x = self.pool(relu(self.conv2(x)))  # bs x 16x 9x 9
        x = x.view(-1, 16 * 9 * 9)          # bs x (16*9*9)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

