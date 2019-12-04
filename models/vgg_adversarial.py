import torch
import torch.nn as nn
from torchvision.models import *
from .utils import load_state_dict_from_url
from collections import OrderedDict

import pdb

__all__ = ['VGG_Adversarial', 'vgg11_adv', 'vgg13_adv', 'vgg16_adv', 'vgg19_adv']

# TODO:
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_adv': None,
    'vgg13_adv': None,
    'vgg16_adv': None,
    'vgg19_adv': None
}

# Height and width (in pixels) of receptive field of each position in the output of
#   the adversary branch, indexed by architecture and number of feature layers
receptive_fields_by_arch = {
    'vgg11_adv': [1, 3, 6, 8, 16, 24, 36, 52, 76],
    'vgg13_adv': [1, 2, 4, 5, 9, 13, 19, 27, 39, 55, 79],
    'vgg16_adv': [1, 2, 4, 5, 9, 13, 15, 23, 31, 39, 51, 67, 83, 107],
    'vgg19_adv': [1, 2, 4, 5, 9, 13, 15, 19, 27, 35, 43, 47, 63, 79, 95, 111, 135]
}


class VGG_Adversarial(nn.Module):

    def __init__(self, arch, feature_layers, init_weights=True):
        super(VGG_Adversarial, self).__init__()
        
        # Create forward architecture from torchvision.models
        task_archs = {'vgg11_adv': vgg11, 'vgg13_adv': vgg13, 'vgg16_adv': vgg16, 'vgg19_adv': vgg19}
        if arch not in task_archs:
            raise ValueError('Invalid VGG architecture: {}'.format(arch))
        self._task_model = task_archs[arch]()
        
        # Set feature_layers
        assert feature_layers > 0 and feature_layers <= (len(self._task_model.features) - 5)/2, 'Invalid number of feature layers: {} (max {})'.format(feature_layers, (len(self._task_model.features) - 5)/2) # VGG has 5 pool layers and 1 ReLU per conv layer
        self.feature_layers = feature_layers
        
        # Make three branches of network
        #   Featurizer and task branch consist of references to task model layers
        #   Adversary model consists of copies of task model layers deeper than featurizer
        self.featurizer, self.task_branch, self.adversary_branch = _make_branches(self._task_model, self.feature_layers, arch)
        
        if init_weights:
            self._initialize_weights()
            
            
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        return self._task_model(x)
    
    
    def adversary(self, x):
        x = self.featurizer(x)
        x = self.adversary_branch(x)
        return x
    

    def load_task_state_dict(self, state_dict):
        self._task_model.load_state_dict(state_dict)
    

    def parameters(self, recurse=True, branch=None): #Override
        if branch == 'task' or branch == 'adversary':
            return [param for name,param in self.named_parameters('',recurse,branch)]
        else:
            return self.parameters(recurse,'task') + self.parameters(recurse,'adversary')


    def named_parameters(self, prefix='', recurse=True, branch=None): #Override
        if branch != 'adversary':
            yield from self.featurizer.named_parameters(prefix+'featurizer',recurse)
            yield from self.task_branch.named_parameters(prefix+'task_branch',recurse)
        if branch != 'task':
            yield from self.adversary_branch.named_parameters(prefix+'adversary_branch',recurse)
        


        
# Given the task model, return the model's three branches: featurizer, task branch, and adversary branch
def _make_branches(task_model, feature_layers, arch):
    # Find index of (feature_layers+1)th conv layer in task_model
    if feature_layers == (len(task_model.features) - 5)/2:
        idx = len(task_model.features)
    else:
        idx = [i for i,layer in enumerate(task_model.features) if isinstance(layer, nn.Conv2d)][feature_layers]
    
    featurizer = nn.Sequential(_name_parameters(list(task_model.features.children())[0:idx]))
    tb_layers = list(task_model.features.children())[idx:] \
                + [task_model.avgpool, nn.Flatten(1)] \
                + list(task_model.classifier.children())
    task_branch = nn.Sequential(_name_parameters(tb_layers, feature_layers+1))
    adversary_branch = _make_adversary_branch(task_branch, feature_layers, arch)

    return featurizer, task_branch, adversary_branch

# Given an interable of layers, return an ordered dictionary of them, named by
#   layer type, numbered starting at (start) 1 and incrementing with each conv or linear layer
def _name_parameters(layers, start=1):
    od = OrderedDict()
    idx = start-1
    for l in layers:
        idx += 1 if isinstance(l, (nn.Conv2d, nn.Linear)) else 0
        name = l.__class__.__name__.lower()
        name = (name[:-2] if name.endswith('2d') else name) + str(idx)
        od[name] = l
    return od

# Given the task branch, replicate it to make the adversary branch, but without
#   dropout or max pool layers, and with 1x1 convolutions instead of linear layers
def _make_adversary_branch(task_branch, feature_layers, arch):
    od = OrderedDict()
    idx = feature_layers
    first_linear = True
    for l in task_branch:
        if isinstance(l, nn.Conv2d):
            idx += 1
            name = 'conv{}'.format(idx)
            layer = nn.Conv2d(in_channels  = l.in_channels,
                              out_channels = l.out_channels,
                              padding      = 0,
                              kernel_size  = 1,
                              bias         = l.bias is not None)
        elif isinstance(l, nn.ReLU):
            name = 'relu{}'.format(idx)
            layer = nn.ReLU(inplace=True)
        elif isinstance(l, nn.Linear):
            idx += 1
            name = 'linear{}'.format(idx)
            layer = nn.Conv2d(in_channels  = 512 if first_linear else l.in_features, # No flattening
                              out_channels = l.out_features,
                              padding      = 0,
                              kernel_size  = 1,
                              bias         = l.bias is not None)
            first_linear = False
        elif isinstance(l, (nn.Dropout, nn.MaxPool2d, nn.Flatten, nn.AdaptiveAvgPool2d)):
            continue
        else:
            raise RuntimeWarning('Unexpected layer in VGG: ' + l.__class__.__name__)
            continue
        od[name] = layer
    # Add final avgpool layer and flatten   TODO: Come back to this. Is this right?
    od['avgpool{}'.format(idx)] = nn.AdaptiveAvgPool2d((1,1))
    od['flatten{}'.format(idx)] = nn.Flatten(1)
    return nn.Sequential(od)


def _vgg_adv(arch, pretrained, task_only, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if 'feature_layers' not in kwargs:
        default_fl = {'vgg11_adv': 5, 'vgg13_adv': 7, 'vgg16_adv': 8, 'vgg19_adv': 9}
        kwargs['feature_layers'] = default_fl[arch]
    model = VGG_Adversarial(arch, **kwargs)
    if pretrained:
        if task_only:
            state_dict = load_state_dict_from_url(model_urls[arch[:-4]], progress=progress)
            model.load_task_state_dict(state_dict)
        else:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model


def vgg11_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial VGG networks are not yet available')
    return _vgg_adv('vgg11_adv', pretrained, task_only, progress, **kwargs)

def vgg13_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial VGG networks are not yet available')
    return _vgg_adv('vgg13_adv', pretrained, task_only, progress, **kwargs)
    
def vgg16_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial VGG networks are not yet available')
    return _vgg_adv('vgg16_adv', pretrained, task_only, progress, **kwargs)

def vgg19_adv(pretrained=False, task_only=False, progress=True, **kwargs):
    if pretrained and not task_only:
        raise NotImplementedError('Pretrained adversarial VGG networks are not yet available')
    return _vgg_adv('vgg19_adv', pretrained, task_only, progress, **kwargs)
    
    
    
    
    
    
