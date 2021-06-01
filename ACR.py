import torch
from torch import nn
from torch.nn.parameter import Parameter

class reduction(nn.Module):
    """
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self,inchannel,outchannel):
        super(reduction, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv3_1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, outchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        p = self.avg_pool(x).view(b, c)
        p = self.fc(p).view(b, 256, 1, 1)
        q = self.conv1(x)
        q = self.bn(q)
        q = self.relu(q)
        q = q * p.expand_as(q)
        k = self.conv3_1(x)
        k = self.relu(k)
        y = q+k
        y = self.conv3_2(y)
        y = self.bn(y)
        y = self.relu(y)
        return y
