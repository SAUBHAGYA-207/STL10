import torch
import torch.nn as nn
import torchvision

def get_stl_resnet18():
    """Modified ResNet-18 optimized for 96x96 STL-10 images."""
    model = torchvision.models.resnet18(num_classes=10)
    # Changing 7x7 conv to 3x3 and removing maxpool to preserve detail for 96x96
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model