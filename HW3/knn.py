import hw3_utils as utils
#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        
        super(Block, self).__init__()
        self.net = nn.Sequential(
        nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_channels),
        torch.nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, kernel_size=3,stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_channels))
           
    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        return (F.relu(self.net(x) + x))


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.block = Block(num_channels)
        self.adapt = nn.AdaptiveAvgPool2d((1))
        self.lin = nn.Linear(num_channels, num_classes)
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        N = x.shape[0]
        print(x.shape)
        out = self.relu(self.norm(self.conv(x)))
        print(out.shape)
        out = self.maxpool(out)
        print(out.shape)
        out = self.block.forward(out)

        print(out.shape)
        out = self.adapt(out)
        print(out.shape)
        out = out.reshape(out.shape[0],-1)
        out = self.lin(out)
        out = out.reshape(N,10)
        print(out.shape)
        return out
##### nearest neighbor problem ###
        
        
def one_nearest_neighbor(X,Y,X_test):
        
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X,Y)
    # return labels for X_test as torch tensor
    return torch.tensor(knn.predict(X_test))
        
        
