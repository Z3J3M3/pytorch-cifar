'''ResNet18 with half channels for CIFAR100 (仅通道数减半，其余逻辑与基线一致)'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础Block保持逻辑不变，仅调整通道数
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 核心修改：所有通道数减半（64→32, 128→64, 256→128, 512→256）
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 32  # 原基线是64，此处减半

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 64→32
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # 64→32
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # 128→64
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2) # 256→128
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2) # 512→256
        self.linear = nn.Linear(256*block.expansion, num_classes)  # 512→256

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 对外暴露的接口，与基线resnet18命名对齐
def resnet18_half(num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
