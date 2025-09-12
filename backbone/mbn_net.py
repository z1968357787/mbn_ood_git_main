'''ResNet in ConfigyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
ConfigreActBlock and ConfigreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

class NormalizeLayer(nn.Module):
    """
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
from abc import *
import torch
import torch.nn as nn

class SimclrLayer(nn.Module):
    """ contrastivie projection layers """
    def __init__(self, Config, last_dim, simclr_dim):
        super(SimclrLayer, self).__init__()

        self.fc1 = nn.Linear(last_dim, last_dim)
        self.relu = nn.ReLU()
        self.last = nn.ModuleList()

        # Initialize embeddings/heads for all the future tasks for convenience.
        # However, it works fine without knowing the number of tasks.
        # Just append a new module when necessary.
        for _ in range(Config.nb_tasks):
            self.last.append(nn.Linear(last_dim, simclr_dim))

        # Configrotect the projection parameters by using hard attention.
        # However, not sure if it's necessary as we don't really use it for inference.
        # Configrotecting/using previously learned parameters possibly encourages faster convergence.
        # TODO: Try without protection.
        self.ec = nn.ParameterList()
        
        for _ in range(Config.nb_tasks):
            self.ec.append(nn.Parameter(torch.randn(1, last_dim)))

        self.gate = torch.sigmoid

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec[t])
        return gc1

    def mask_out(self, out, mask):
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, features, s=1):
        gc1 = self.mask(t, s=s)

        out = self.fc1(features)
        out = self.relu(out)
        out = self.mask_out(out, gc1)
        out = self.last[t](out)
        return out, gc1

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, Config, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.Config = Config

        # Standard classifier without ensemble. This is only for reference during feature training.
        # Not necessary for testing. Does not affect the final result with/without it during training.
        self.linear = nn.ModuleList()
        for _ in range(Config.nb_tasks):
            self.linear.append(nn.Linear(last_dim, num_classes))

        # Contrastive projection
        self.simclr_layer = SimclrLayer(Config, last_dim, simclr_dim)

        # Independent ensemble classifier
        self.joint_distribution_layer = nn.ModuleList()
        for _ in range(Config.nb_tasks):
            self.joint_distribution_layer.append(nn.Linear(last_dim, 4 * num_classes))

    @abstractmethod
    def penultimate(self, t, inputs, s, all_features=False):
        pass

    def forward(self, t, inputs, s=1, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        features, masks = self.penultimate(t, inputs, s)

        output = self.linear[t](features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            out, gc1 = self.simclr_layer(t, features, s)
            masks.append(gc1)
            _aux['simclr'] = out

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer[t](features)

        if _return_aux:
            return output, _aux, masks

        return output, masks

class Downsample(nn.Module):
    def __init__(self, Config, stride, in_planes, expansion, planes):
        super(Downsample, self).__init__()
        self.identity = True
        self.downsample = nn.Sequential()

        if stride != 1 or in_planes != expansion*planes:
            self.identity = False
            self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.bn1 = nn.ModuleList()
            for _ in range(Config.nb_tasks):
                self.bn1.append(nn.BatchNorm2d(expansion*planes))

    def forward(self, t, x):
        if self.identity:
            out = self.downsample(x)
        else:
            out = self.conv1(x)
            out = self.bn1[t](out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, Config, in_planes, planes, stride=1, pooling=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.ModuleList()
        for _ in range(Config.nb_tasks):
            self.bn1.append(nn.BatchNorm2d(planes))
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.ModuleList()
        for _ in range(Config.nb_tasks):
            self.bn2.append(nn.BatchNorm2d(planes))
        # self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = Downsample(Config, stride, in_planes, self.expansion, planes)

        self.gate = torch.sigmoid
        self.ec1 = nn.ParameterList()
        self.ec2 = nn.ParameterList()
        for _ in range(Config.nb_tasks):
            self.ec1.append(nn.Parameter(torch.randn(1, planes)))
            self.ec2.append(nn.Parameter(torch.randn(1, planes)))
        # self.ec1 = nn.Embedding(5, planes)
        # self.ec2 = nn.Embedding(5, planes)

        self.pooling = pooling


    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec1[t])
        gc2 = self.gate(s * self.ec2[t])
        return [gc1, gc2]

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, x, msk, s):
        masks = self.mask(t, s=s)
        gc1, gc2 = masks

        msk.append(masks)

        out = F.relu(self.bn1[t](self.conv1(x)))
        out = self.mask_out(out, gc1)

        out = self.bn2[t](self.conv2(out))
        out += self.downsample(t, x)
        out = F.relu(out)

        if self.pooling:
            out = F.avg_pool2d(out, 4)

        out = self.mask_out(out, gc2)
        return t, out, msk, s


# class ConfigreActBlock(nn.Module):
#     '''Configre-activation version of the BasicBlock.'''
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(ConfigreActBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out)
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += shortcut
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ConfigreActBottleneck(nn.Module):
#     '''Configre-activation version of the original Bottleneck module.'''
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(ConfigreActBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.bn3 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(x))
#         shortcut = self.shortcut(out)
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = self.conv3(F.relu(self.bn3(out)))
#         out += shortcut
#         return out


class ResNet(BaseModel):
    def __init__(self, Config, block, num_blocks, num_classes=10):
        if 'cifar10' in Config.dataset and 'cifar100' not in Config.dataset:
            init_dim, last_dim = 64, 512 # last_dim = init_dim * 8
        elif 'cifar100' in Config.dataset or 'tinyImagenet' in Config.dataset:
            init_dim, last_dim = 128, 1024
        super(ResNet, self).__init__(last_dim, Config, num_classes)

        self.in_planes = init_dim
        self.last_dim = last_dim

        self.normalize = NormalizeLayer()

        self.conv1 = conv3x3(3, init_dim)
        
        # Initialize all the task specific parameters for convenience.
        # No need to know the number of tasks as we can add new ones when necessary.
        self.bn1 = nn.ModuleList()
        for _ in range(Config.nb_tasks):
            self.bn1.append(nn.BatchNorm2d(init_dim))

        self.ec0 = nn.ParameterList()
        for _ in range(Config.nb_tasks):
            self.ec0.append(nn.Parameter(torch.randn(1, init_dim)))

        self.gate = torch.sigmoid

        self.layer1 = self._make_layer(Config, block, init_dim, num_blocks[0], stride=1, pooling=False)
        self.layer2 = self._make_layer(Config, block, init_dim * 2, num_blocks[1], stride=2, pooling=False) # LAYER2.0.SHORTCUT
        self.layer3 = self._make_layer(Config, block, init_dim * 4, num_blocks[2], stride=2, pooling=False) # LAYER3.0.SHORTCUT
        self.layer4 = self._make_layer(Config, block, last_dim, num_blocks[3], stride=2, pooling=True) # LAYER4.0.SHORTCUT

    def _make_layer(self, Config, block, planes, num_blocks, stride, pooling=False):
        strides = [stride] + [1]*(num_blocks-1)
        pooling_ = False
        layers = nn.ModuleList()
        for i, stride in enumerate(strides):
            if i == len(strides) - 1:
                pooling_ = pooling
            layers.append(block(Config, self.in_planes, planes, stride, pooling_))
            self.in_planes = planes * block.expansion
        return layers

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec0[t])
        return gc1

    def mask_out(self, out, mask):
        mask = mask.view(1, -1, 1, 1)
        out = out * mask.expand_as(out)
        return out

    def penultimate(self, t, x, s=1):
        out_list = []
        msk = []

        gc0 = self.mask(t, s=s)
        msk.append(gc0)

        x = self.normalize(x)
        x = self.conv1(x)
        x = self.bn1[t](x)
        x = F.relu(x)
        x = self.mask_out(x, gc0)

        for op in self.layer1:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer1(out)

        for op in self.layer2:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer2(out)

        for op in self.layer3:
            t, x, msk, s = op(t, x, msk, s)
        # out = self.layer3(out)

        for op in self.layer4:
            t, x, msk, s = op(t, x, msk, s) # the output x is (100, 512, 1, 1)
        # out = self.layer4(out)

        # out = F.avg_pool2d(x, 4)
        out = x.view(x.size(0), -1)

        return out, list(itertools.chain(*msk))


def ResNet18(Config, num_classes):
    return ResNet(Config, BasicBlock, [2,2,2,2], num_classes=num_classes)
