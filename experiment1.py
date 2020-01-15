import os
import random
import shutil
import time
import warnings
import numpy as np
import pdb
import time

import torch
import torch.nn asnn
import torch.optim as optim
import torchvision.transforms as transforms

import models
from adversarial_optim import AdversarialWrapper as optim_wrapper
import mnist_mosaic
from pytorch_utils import MnistMosaicDataset
from utils import AverageMeter, ProgressMeter

def main():
    
    # Load model
    print('Loading model...')
    model = models.vgg13_adv()
    
    # Make optimizers
    print('Making optimizers...')
    task_optim = optim.SGD(model.task_parameters(), lr = 0.001, momentum = 0.0)
    adv_optim = optim.SGD(model.adversary_parameters(), lr = 0.01, momentum = 0.0)
    optimizer = optim_wrapper(task_optim, adv_optim, 1)
    criterion = nn.CrossEntropyLoss()
    
    # Access data
    print('Accessing data...')
    train_dataset = MnistMosaicDataset(
                    'mnist_data/mosaic_correlated_train.npz',
                    transform=transforms.ToTensor(),
                    label_only=True)
    test_dataset = MnistMosaicDataset(
                    'mnist_data/mosaic_correlated_test.npz',
                    transform=transforms.ToTensor(),
                    label_only=True)

    # Make data loaders
    print('Making data loaders...')
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=64, shuffle=True, num_workers=4,
                    pin_memory=False, sampler=None)
    test_loader = torch.utils.data.DataLoade(
                    test_dataset, batch_size=64, shuffle=False, num_workers=4,
                    pin_memory=False, sampler=None)
    
    # Pretrain task branch
    print('Pretraining task branch...')
    optimizer.mode('task')
    for epoch in range(0, 10):
        train_one_epoch(train_loader, model, optimizer, criterion, epoch) #TODO: add other args
    
    # Pretrain adversary branch
    print('Pretraining adversary branch...')
    optimizer.mode('adversary')
    for epoch in range(0, 10):
        train_one_epoch(train_loader, model, optimizer, criterion, epoch) #TODO: add other args
        
    # Evaluate and checkpoint
    acc1 = evaluate(test_loader, model, criterion) #TODO: add other args
    best_acc1 = acc1
    save_checkpoint({
        'epoch': 0,
        'arch': 'vgg13_adv',
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict()},
        True)
    
    # Train network
    print('Training network...')
    optmiizer.mode('train')
    for epoch in range(0, 20):
        train_one_epoch(train_loader, model, optimizer, criterion, epoch) #TODO: add other args
        
        acc1 = evaluate(test_loader, model, criterion) #TODO: add other args
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch+1,
            'arch': 'vgg13_adv',
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()},
            is_best)
    
def train_one_epoch(train_loader, model, optimizer, criterion, epoch):
    
    batch_time = AverageMeter('Time', ':6.3f')
    task_losses = AverageMeter('Task Loss', ':.3e')
    adv_losses = AverageMeter('Adv Loss', ':.3e')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, task_losses, adv_losses, top1, top5],
        prefix="Ep: [{}]".format(epoch))

    # Switch to training mode
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        # Compute outputs and losses
        (task_output,adv_output) = model(images)
        task_loss = criterion(task_output, target)
        adv_loss = criterion(adv_output, target)
        combined_loss = task_loss - adv_loss
        
        # Choose relevant loss type
        if optimizer.step_type() == 'adversary':
            loss = adv_loss
        elif optimizer.step_type() == 'task':
            loss = task_loss if optimizer._mode == 'task' else combined loss
        
        # Measure accuracy and record losses
        acc1, acc5 = accuracy(task_output, target, topk=(1,5))
        task_losses.update(task_loss.item(), images.size(0))
        adv_losses.update(adv_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Compute gradients and take step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Automatically alternate task/adversary if mode == 'train'
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            progress.display(i)


def evaluate(test_loader, model, criterion):
    
    batch_time = AverageMeter('Time', ':6.3f')
    task_losses = AverageMeter('Task Loss', ':.3e')
    adv_losses = AverageMeter('Adv Loss', ':.3e')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, task_losses, adv_losses, top1, top5],
        prefix="Ep: [{}]".format(epoch))
    
    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        
        for i, (images, target) in enumerate(test_loader):
            
            # Compute outputs and losses
            task_output, adv_output = model(images)
            task_loss = criterion(task_output, target)
            adv_loss = criterion(adv_output, target)
            
            # Measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            task_losses.update(task_loss.item(), images.size(0))
            adv_losses.update(adv_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                progress.display(i)
            
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# From <https://github.com/pytorch/examples/blob/master/imagenet/main.py>
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    





