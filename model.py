import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fc1 = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc3 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        
        x = self.conv4(x)
        
        x = self.conv5(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x