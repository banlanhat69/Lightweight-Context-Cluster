from __future__ import annotations

from torch import nn
from torchvision import models


def resnet18_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def mobilenet_v2_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    model.features[0][0].stride = (1, 1)
    return model


def shufflenet_v2_x1_0_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    model.conv1[0].stride = (1, 1)
    model.maxpool = nn.Identity()
    return model
