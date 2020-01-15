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
    

