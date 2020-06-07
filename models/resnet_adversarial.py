import torch
import torch.nn as nn
from torchvision.models import *
from torchvision.models.resnet import BasicBlock,Bottleneck
from .utils import load_state_dict_from_url
from collections import OrderedDict
from .local_adversary_nets import LocalAdversaryNet

import pdb

__all__ = ['ResNet_Adversarial', 'resnet18_adv', 'resnet34_adv', 'resnet50_adv', 'resnet101_adv', 'resnet152_adv']

model_urls = {
    'resnet18_adv': None,
    'resnet34_adv': None,
    'resnet50_adv': None,
    'resnet101_adv': None,
    'resnet152_adv': None
}


class ResNet_Adversarial(LocalAdversaryNet):

    def __init__(self, arch, feature_layers, init_weights=True, task_state_dict=None):
        task_archs = {'resnet18_adv': resnet18,
                      'resnet34_adv': resnet34,
                      'resnet50_adv': resnet50,
                      'resnet101_adv': resnet101,
                      'resnet152_adv': resnet152}
        assert arch in task_archs, 'Invalid ResNet architecture name: {}'.format(arch)
        assert feature_layers > 0 and feature_layers < 6, 'Invalid number of feature layers: {} (min 1, max 5)'.format(feature_layers)

        task_model = task_archs[arch]()

        self.inplanes = self.outplanes = 0
        super(ResNet_Adversarial, self).__init__(task_model, feature_layers, self._layer_iter, self._convert, self._touch)
        if init_weights:
            self._initialize_weights()
        if task_state_dict is not None:
            task_model.load_state_dict(task_state_dict)



    def _touch(self, layer, idx):
        self.inplanes = self.outplanes
        if isinstance(layer[0], nn.Conv2d):
            self.outplanes = layer[0].out_channels
        elif isinstance(layer[0], nn.Linear):
            self.outplanes = layer[0].out_features
        elif isinstance(layer[0], (BasicBlock, Bottleneck)):
            self.outplanes = self.inplanes * layer[0].expansion

    def _layer_iter(self, task_model):
        yield [task_model.conv1, task_model.bn1, task_model.relu, task_model.maxpool]
        yield [task_model.layer1]
        yield [task_model.layer2]
        yield [task_model.layer3]
        yield [task_model.layer4, task_model.avgpool, nn.Flatten(1)]
        yield [task_model.fc]

    def _convert(self, task_layer):
        adv_layer = []
        for module in task_layer:
            if isinstance(module, nn.Sequential):
                adv_layer += self._convert([m for m in module])
            elif isinstance(module, nn.Conv2d):
                adv_layer.append(nn.Conv2d(in_channels = module.in_channels,
                                 out_channels = module.out_channels,
                                 padding = module.padding,
                                 kernel_size = 1,
                                 bias = module.bias is not None))
            elif isinstance(module, nn.ReLU):
                adv_layer.append(nn.ReLU(inplace=True))
            elif isinstance(module, nn.Linear):
                adv_layer.append(nn.Conv2d(in_channels = self.inplanes,
                                 out_channels = module.out_features,
                                 padding = 0,
                                 kernel_size = 1,
                                 bias = module.bias is not None))
            elif isinstance(module, (nn.Dropout, nn.MaxPool2d, nn.Flatten, nn.AdaptiveAvgPool2d)):
                pass
            elif isinstance(module, BasicBlock):
                adv_layer.append(BasicBlock(inplanes = self.inplanes,
                                            planes = self.outplanes,
                                            stride = module.stride,
                                            downsample=module.downsample,
                                            groups = 1,
                                            base_width = 64,
                                            dilation = 1,
                                            norm_layer = module.bn1.__class__))
            elif isinstance(module, Bottleneck):
                adv_layer.append(Bottleneck(inplanes = self.inplanes,
                                            planes = self.outplanes,
                                            stride = module.stride,
                                            downsample = module.downsample,
                                            groups = module.conv2.groups,
                                            base_width = self.task_model.base_width,
                                            dilation = module.conv2.dilation,
                                            norm_layer = module.bn1.__class__))
            else:
                raise ValueError('Unexpected class in ResNet architecture: {}'.format(module.__class__))
        return adv_layer

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)


def _resnet_adv(arch, pretrained, task_only, progress, **kwargs):
    if pretrained and task_only:
        kwargs['init_weights'] = True
        state_dict = load_state_dict_from_url(model_urls[arch[:-4]], progress=progress)
        kwargs['task_state_dict'] = state_dict
    elif pretrained and not task_only:
        kwargs['init_weights'] = False

    if 'feature_layers' not in kwargs or kwargs['feature_layers'] is None:
        kwargs['feature_layers'] = 3 #TODO: Investigate a better default
    model = ResNet_Adversarial(arch, **kwargs)

    if pretrained and not task_only:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial ResNet not yet available')
    return _resnet_adv('resnet18_adv', pretrained, task_only, progress, **kwargs)

def resnet34_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial ResNet not yet available')
    return _resnet_adv('resnet34_adv', pretrained, task_only, progress, **kwargs)

def resnet50_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial ResNet not yet available')
    return _resnet_adv('resnet50_adv', pretrained, task_only, progress, **kwargs)

def resnet101_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial ResNet not yet available')
    return _resnet_adv('resnet101_adv', pretrained, task_only, progress, **kwargs)

def resnet152_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial ResNet not yet available')
    return _resnet_adv('resnet152_adv', pretrained, task_only, progress, **kwargs)

