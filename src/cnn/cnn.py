import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn.SRM_filters import get_filters


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # The SRM layer is now the first layer.
        # It takes 3 input channels and outputs 30 feature maps.
        self.conv0 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv0.weight = nn.Parameter(get_filters(), requires_grad=False)  # Fixed Weights (SRM)

        # CRITICAL FIX: LRN size must be 30 to cover all feature maps.
        self.lrn = nn.LocalResponseNorm(30)  # <-- FIXED TO 30
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(30, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv8 = nn.Conv2d(16, 16, 3, padding=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # New first layer (SRM)
        x = F.relu(self.conv0(x))

        x = self.lrn(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.lrn(x)
        x = self.pool(x)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)