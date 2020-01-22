import torch
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import mosaic

# Example instantiation of dataset:
# train_set = mosaic_mnist.MnistMosaicDataset(
#               'consistent_train',
#               transform = transforms.Compose([
#                   transforms.ToTensor(),
#                   mosaic_mnist.grayscale2color]),
#               label_only=True)



class MnistMosaicDataset(data.Dataset):
    
    def __init__(self, fname, transform=None, label_only=True):
        self._x, self._y = mosaic.load(fname, label_only=label_only, dense=False)
        self.transform = transform
    
    def __len__(self):
        return self._x.shape[0]
    
    def __getitem__(self, index):
        img = self._x[index,:].todense().getA().reshape([224,224,1])
        if self.transform:
            img = self.transform(img)
        return img, int(self._y[index])


# Takes in a [1, H, W] or [N, 1, H, W] Tensor and returns a [3, H, W] or [N, 3, H, W] Tensor by 
#   viewing the underlying data as the same channel 3 times.   
def grayscale2color(img):
    shape = list(img.shape)
    shape[-3] = 3
    return img.expand(*shape)
