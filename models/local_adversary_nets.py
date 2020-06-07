import torch
import torch.nn as nn
from collections import OrderedDict

import pdb



class LocalAdversaryNet(nn.Module):

    '''
    Inputs:
        task_model: A torch.nn.Module, the model we want to form a local adversary with
        feature_layers: Number of layers of the task model that should be shared between task and
                adversary branches
        layer_iter: A generator that takes in the task model and sequentially yields a list
                of Modules representing one "layer" in the model. These are the "layers" referenced
                by input feature_layers
        convert: Method that takes in a layer (as output from layer_iter) and returns a layer
                of new Modules, corresponding to the adversarial version of the task layer. For example,
                a simple method replaces:
                    conv2d -> 1x1 conv2d
                    linear -> 1x1 conv2d
                    relu -> relu
                    pooling functions -> <nothing>
        touch (Optional): Method called on each layer before adding it to a branch or converting it.
                Takes in the layer and index as inputs.
    '''
    def __init__(self, task_model, feature_layers, layer_iter, convert, touch=None):
        super(LocalAdversaryNet, self).__init__()
        
        self.feature_layers = feature_layers
        featurizer = []
        task_branch = []
        adversary_branch = []
        idx = 0

        for layer in layer_iter(task_model):
            if touch is not None:
                touch(layer, idx)
            if idx < feature_layers:
                featurizer += layer
            else:
                task_branch += layer
                adversary_branch += convert(layer)
            idx += 1

        adversary_branch.append(nn.AdaptiveAvgPool2d((1,1))) # Aggregate local predictions
        adversary_branch.append(nn.Flatten(1)) # Squeeze dimensions

        self.featurizer = nn.Sequential(*featurizer)
        self.task_branch = nn.Sequential(*task_branch)
        self.adversary_branch = nn.Sequential(*adversary_branch)

    def forward(self, x):
        features = self.featurizer(x)
        return (self.task_branch(features), self.adversary_branch(features))

'''
    Example class that converts an nn.Sequential of conv, linear, relu, and pool modules
    into the equivalent adversarial version.
'''
class SequentialAdversary(LocalAdversaryNet):

    def __init__(self, task_model, feature_layers):
        assert isinstance(task_model, nn.Sequential), 'task_model must be a torch.nn.Sequential'
        self.inplanes = 0
        self.outplanes = 0
        super(SequentialAdversary, self).__init__(task_model, feature_layers, self._layer_iter, self._convert, self._touch)
        delattr(self, 'inplanes')
        delattr(self, 'outplanes')


    def _layer_iter(self, task_model):
        layer = None
        for module in task_model:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if layer is not None:
                    yield layer
                layer = []
            layer.append(module)
        if len(layer) > 0:
            yield layer

    def _touch(self, layer, idx):
        self.inplanes = self.outplanes
        if isinstance(layer[0], nn.Conv2d):
            self.outplanes = layer[0].out_channels
        elif isinstance(layer[0], nn.Linear):
            self.outplanes = layer[0].out_features


    def _convert(self, layer):
        new_layer = []
        for module in layer:
            new_module = self._convert_one(module)
            if new_module is not None:
                new_layer.append(new_module)
        return new_layer

    def _convert_one(self, module):
        if isinstance(module, nn.Conv2d):
            return nn.Conv2d(in_channels = module.in_channels,
                             out_channels = module.out_channels,
                             padding = module.padding,
                             kernel_size = 1,
                             bias = module.bias is not None)
        elif isinstance(module, nn.ReLU):
            return nn.ReLU(inplace=True)
        elif isinstance(module, nn.Linear):
            return nn.Conv2d(in_channels = self.inplanes,
                             out_channels = module.out_features,
                             padding = 0,
                             kernel_size = 1,
                             bias = module.bias is not None)
        elif isinstance(module, (nn.Dropout, nn.MaxPool2d, nn.Flatten, nn.AdaptiveAvgPool2d)):
            return None
        else:
            raise RuntimeWarning('Unexpected layer in Sequential model: ' + module.__class__.__name__)
            return None

