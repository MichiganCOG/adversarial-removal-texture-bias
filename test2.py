import torch
import torch.nn as nn
import numpy as np
from models import *
import pdb

def zero_all_but_one_hook(module, grad_in, grad_out):
    newgrad = torch.zeros(grad_in[0].size(),dtype=torch.double)
    newgrad.data[:,:,0,0] = 1.0
    return (newgrad,
            torch.zeros(grad_in[1].size(),dtype=torch.double),
            torch.zeros(grad_in[2].size(),dtype=torch.double))

def receptive_field(input_grad):
    input_grad = np.any(input_grad, axis=(0,1))
    shp = input_grad[input_grad].shape
    return int(np.sqrt(shp[0]))


trials = {'vgg11_adv': 8, 'vgg13_adv': 10, 'vgg16_adv': 13, 'vgg19_adv': 16}
results = {a: [0]*b for a,b in trials.items()}
dummy_img = torch.autograd.Variable(torch.from_numpy(np.ones((1,3,244,244))), requires_grad=True)
dummy_label = torch.from_numpy(np.ones((1,), dtype=np.int64))
criterion = nn.CrossEntropyLoss()


for arch in trials.keys():
    print('{}: '.format(arch), end='')
    for feature_layers in range(trials[arch]):
        model = globals()[arch](feature_layers=feature_layers+1, init_weights=True)
        model.double()
        hook = model.adversary_branch[0].register_backward_hook(zero_all_but_one_hook)

        if dummy_img.grad is not None:
            dummy_img.grad *= 0.0
        loss = criterion(model.adversary(dummy_img), dummy_label)
        loss.backward()
        
        #pdb.set_trace()
        results[arch][feature_layers] = receptive_field(dummy_img.grad.detach().numpy())
        print('{} '.format(results[arch][feature_layers]), end='')
    print('')

pdb.set_trace()

'''        Feature Layers
Arch.      (0) 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16

vgg11_adv  (1) 3   6   8  16  24  36  52  76
vgg13_adv  (1) 2   4   5   9  13  19  27  39  55  79
vgg16_adv  (1) 2   4   5   9  13  15  23  31  39  51  67  83 107
vgg19_adv  (1) 2   4   5   9  13  15  19  27  35  43  47  63  79  95 111 135
'''
    
