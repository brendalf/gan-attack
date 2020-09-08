import torch.nn as nn
import torch.nn.functional as F


class Cifar10Custom(nn.Module):
    """Cifar10 target model"""

    def __init__(self):
        """Cifar10 Builder."""
        super(Cifar10Custom, self).__init__()

        #Conv Layer Block 1:
        self.conv1_1  = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batch1_1 = nn.BatchNorm2d(32)
        self.relu1_1  = nn.ReLU(inplace=True)
        self.conv1_2  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu1_2  = nn.ReLU(inplace=True)
        self.maxp1_1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer block 2
        self.conv2_1  = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch2_1 = nn.BatchNorm2d(128)
        self.relu2_1  = nn.ReLU(inplace=True)
        self.conv2_2  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu_2_2 = nn.ReLU(inplace=True)
        self.maxp2_1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2_1  = nn.Dropout2d(p=0.05)

        # Conv Layer block 3
        self.conv3_1  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch3_1 = nn.BatchNorm2d(256)
        self.relu3_1  = nn.ReLU(inplace=True)
        self.conv3_2  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2  = nn.ReLU(inplace=True)
        self.maxp_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x, batch_norm=True):
        """Perform forward."""
        
        #Conv Layer Block 1:
        x = self.conv1_1(x)
        if batch_norm: x = self.batch1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.maxp1_1(x)

        # Conv Layer block 2
        x = self.conv2_1(x)
        if batch_norm: x = self.batch2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu_2_2(x)
        x = self.maxp2_1(x)
        x = self.drop2_1(x)

        # Conv Layer block 3
        x = self.conv3_1(x)
        if batch_norm: x = self.batch3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.maxp_3_1(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x