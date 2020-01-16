import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import mnist_mosaic as mm


class MnistMosaicDataset(data.Dataset):
    
    def __init__(self, fname, transform=None, label_only=True):
        self._x, self._y = mm.load(fname, label_only=label_only, dense=False)
        self.transform = transform
    
    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, index):
        img = self._x[index,:].todense().getA().reshape([224,224,1])
        if self.transform:
            img = self.transform(img)
        return img, self._y[index]


# Takes in a [N, 1, H, W] Tensor and returns a [N, 3, H, w] Tensor by viewing the underlying
#    data as the same channel 3 times. New tensor is not writeable.    
def grayscale2color(img):
    shape = list(img.detach().numpy().shape)
    shape[1] = 3
    strides = list(img.detach().numpy().strides)
    strides[1] = 0 # Honestly don't ever do this
    return torch.tensor(np.lib.stride_tricks.as_strided(
            img.data,
            shape = tuple(shape),
            strides = tuple(strides),
            writeable = False),
            requires_grad = False)
