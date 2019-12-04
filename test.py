import torch
import torch.nn as nn
import numpy as np
from models import *
import torch.optim as optim
from adversarial_optim import AdversarialWrapper as optim_wrapper
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pdb

print('Loading model...')
model = vgg11_adv(pretrained=True, task_only=True, feature_layers=6)
model = torch.nn.DataParallel(model)

print('Making optimizers...')
task_optim = optim.SGD(model.parameters(branch='task'), lr = 0.0001, momentum = 0.0)
adv_optim = optim.SGD(model.parameters(branch='adversary'), lr = 0.001, momentum = 0.0)
optimizer = optim_wrapper(task_optim, adv_optim)
criterion = nn.CrossEntropyLoss()

traindir = '/z/dat/ImageNet_2012/train'
valdir = '/z/dat/ImageNet_2012/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

print('Accessing training data...')
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))

print('Making data loader...')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, sampler=None) #TODO: 256

print('Training:')
for i, (images, target) in enumerate(train_loader):
    
    task_output = model.forward(images)
    adv_output = model.adversary(images)
    
    task_loss = criterion(task_output, target)
    adv_loss = criterion(adv_output, target)
    combined_loss = task_loss - adv_loss
    
    print(' [Batch {}]\ttask {}\tadv {}'.format(i,task_loss,adv_loss))
    
    combined_loss.backward()
    optimizer.zero_grad()
    optimizer.step() # Will alternate automatically between task & adversary
    
    


'''
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])),
    batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
'''



'''
def train(train_loader, model, criterion, optimizer, epoch, args):
    
    model.train()
    
    for i, (images,target) in enumerate(train_loader):
        
        task_output = model.forward(images)
        adv_output = model.adversary(images)
        
        task_loss = criterion(output, target)
'''
    
    
    
    
    
    
    





