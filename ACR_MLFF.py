import os
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import config
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
from ACR import reduction

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ACR_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21, scale=1):
        self.inplanes = 64
        super(ACR_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ###########################################
        self.C5_P5 = reduction(2048,256)
        self.C4_P4 = reduction(1024,256)
        self.C3_P3 = reduction(512,256)
        # self.C2_P2 = reduction(256,256)
        self.down28 = nn.AdaptiveAvgPool2d((28,28))
        self.down14 = nn.AdaptiveAvgPool2d((14,14))
        self.down7 = nn.AdaptiveAvgPool2d((7,7))
        # ############################################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        c2 = h
        h = self.layer2(h)
        c3 = h
        h = self.layer3(h)
        c4 = h
        h = self.layer4(h)
        c5 = h
        # using ACR block
        c5 = self.C5_P5(c5)
        c4 = self.C4_P4(c4)
        c3 = self.C3_P3(c3)
        # c2 = self.C2_P2(c2)

        out1 = self.down7(c2)
        out2 = self.down7(c3)
        out3 = self.down7(c4)
        out4 = self.down7(c5)

        out = torch.cat((out1,out2,out3,out4),1)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return out


def acr_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ACR_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


class ACR_MLFF(nn.Module):
    def __init__(self):
        super(ACR_MLFF, self).__init__()

        # self.backbone = resnet50(pretrained=True, num_classes=10)
        self.backbone = self._get_backbone()
        self.fc = nn.Linear(1024, config.NUM_CLASSES)
        # self.softmax = nn.LogSoftmax(dim=1)
    def _get_backbone(self):
        backbone = acr_resnet50(pretrained=True, num_classes=config.NUM_CLASSES)

        # for param in backbone.layer1.parameters():
        #     param.requires_grad = False
        # for param in backbone.layer2.parameters():
        #     param.requires_grad = False
        # for param in backbone.layer3.parameters():
        #     param.requires_grad = False
        # for param in backbone.layer4.parameters():
        #     param.requires_grad = False
        return backbone

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1,1024)
        # out = out.view(-1, 21 * 124 * 124)
        # out = self.fc(out)
        # print('output shape',out.shape)
        out = self.fc(out)
        return out