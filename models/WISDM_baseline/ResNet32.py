import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, channels=1, num_classes=18):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1   = nn.Conv2d(channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(self.in_planes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1  = self._make_layer(block, 64, layers[0])
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Sequential(
            nn.Linear(200, 32 * 32),
            nn.GELU()
        )

        self.fc2      = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = self.fc1(x)
        x = x.view(x.size()[0], x.size()[1], 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


