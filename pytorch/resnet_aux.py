import torch
from torch import nn
from torch.nn import functional
from torchvision.models.resnet import ResNet as _ResNet
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import load_state_dict_from_url

from aux_nets import InceptionAux

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)

class Loss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', cats=None):
        nn.CrossEntropyLoss.__init__(
            self,
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        cuda_cats = []
        for cat in cats:
            cuda_cats.append(torch.from_numpy(cat).long().cuda())
        self.cats = cuda_cats

    def forward(self, sample, **kwargs):
        l = nn.CrossEntropyLoss.forward(
            self,
            sample['logits'],
            sample['label'].squeeze()
        )
        for i in range(len(self.cats)):
            l += nn.CrossEntropyLoss.forward(
                self,
                sample['aux'+str(i+1)],
                self.cats[i][sample['label']].squeeze()
            )
        return l


class ResNet(_ResNet):
    def __init__(self, block, layers, num_coarse_classes, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, mean=None, std=None):
        _ResNet.__init__(
            self,
            block, layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )

        if mean is None:
            self.register_buffer(
                'mean', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)/255
            )
        else:
            self.register_buffer(
                'mean', torch.tensor(mean).view(1, -1, 1, 1)
            )
        if std is None:
            self.register_buffer(
                'std', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)/255
            )
        else:
            self.register_buffer(
                'std', torch.tensor(std).view(1, -1, 1, 1)
            )


        self.aux1 = InceptionAux(in_channels=4*64, num_coarse_classes=num_coarse_classes[0], pool=2)
        self.aux2 = InceptionAux(in_channels=4*128, num_coarse_classes=num_coarse_classes[1], pool=1)
        self.aux3 = InceptionAux(in_channels=4*256, num_coarse_classes=num_coarse_classes[2])

    def forward(self, sample):
        with torch.no_grad():
            x = sample['image']
            x = x.float()
            x = x.sub(self.mean).div(self.std)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        y1 = self.aux1(x)

        x = self.layer2(x)
        y2 = self.aux2(x)

        x = self.layer3(x)
        y3 = self.aux3(x)

        conv_output = self.layer4(x)

        x = self.avgpool(conv_output)
        features = torch.flatten(x, 1)

        x = self.fc(features)

        return dict(sample, logits=x, aux1=y1, aux2=y2, aux3=y3, output=x)

def _resnet(arch, block, layers, pretrained, progress, num_coarse_classes, **kwargs):
    model = ResNet(block, layers, num_coarse_classes, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50_aux(num_coarse_classes, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, False, num_coarse_classes, **kwargs)
