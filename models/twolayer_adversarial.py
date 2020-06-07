import torch
import torch.nn as nn

__all__=['twolayer_adv']

class TwoLayer_Adversarial(nn.Module):

    def __init__(self, kernel_size, num_kernels, img_channels=3):
        super(TwoLayer_Adversarial, self).__init__()

        self.feature_layers = 1

        kernel_size = (kernel_size,kernel_size) if isinstance(kernel_size, int) else kernel_size

        conv_out = num_kernels * (224 - kernel_size[0] + 1) * (224 - kernel_size[1] + 1)

        self.featurizer = nn.Sequential(
                nn.Conv2d(in_channels  = img_channels,
                          out_channels = num_kernels,
                          padding      = 0,
                          kernel_size  = kernel_size,
                          bias         = True),
                nn.ReLU(inplace=True)
        )

        self.task_branch = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(conv_out, 10, True)
        )

        self.adversary_branch = nn.Sequential(
                nn.Conv2d(in_channels  = num_kernels,
                          out_channels = 10,
                          padding      = 0,
                          kernel_size  = 1,
                          bias         = True),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(1)
        )

        
        nn.init.kaiming_normal_(self.featurizer[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.featurizer[0].bias, 0)
        nn.init.normal_(self.task_branch[1].weight, 0, 0.01)
        nn.init.constant_(self.task_branch[1].bias, 0)
        nn.init.kaiming_normal_(self.adversary_branch[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.adversary_branch[0].bias, 0)

    def forward(self, x):
        features = self.featurizer(x)
        return (self.task_branch(features), self.adversary_branch(features))

    def task_parameters(self):
        return [p for n,p in self.named_parameters() if 'featurizer' in n or 'task_branch' in n]

    def adversary_parameters(self):
        return [p for n,p in self.named_parameters() if 'adversary_branch' in n]

# Called by main.py, dummy arguments are to match VGG function signature
def twolayer_adv(feature_layers=False, pretrained=False, task_only=False):
    return TwoLayer_Adversarial((31,31), 64, 3)
