import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from adversarial_optim import AdversarialWrapper as optim_wrapper
from utils import AverageMeter, ProgressMeter


# All acceptable model architecture names
model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith('__')
                    and callable(models.__dict__[name]))

best_acc1=0

def main(args):

    print('Args:')
    print(args)

    print('Loading model...')
    pretrained = args.pretrained == 'both' or args.pretrained == 'task'
    task_only = args.pretrained == 'task'
    model = models.__dict__[args.arch](feature_layers=args.feature_layers,
                                       pretrained=pretrained,
                                       task_only=task_only)
    if args.precision == 'double':
        model.double()
    elif args.precision == 'half':
        model.half()

    if args.use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)


    print('Making optimizers...')
    task_optim = optim.SGD(model.task_parameters(), lr=args.lr, momentum=args.momentum)
    adv_optim = optim.SGD(model.adversary_parameters(), lr=args.adv_lr, momentum=args.adv_momentum)

    optimizer = optim_wrapper(task_optim, adv_optim, args.eta)
    criterion = nn.CrossEntropyLoss()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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
            
    print('Accessing validation data...')
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]))
            
    print('Making data loaders...')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=args.pretrained != 'none', sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=args.pretrained != 'none', sampler=None)
        
    if args.pretrain_epochs > 0:
        print('Pretraining task branch:')
        optimizer.mode('task')
        for epoch in range(0, args.pretrain_epochs):
            # Train one epoch
            train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)

    if args.adv_pretrain_epochs > 0:
        print('Pretraining adversary branch:')
        optimizer.mode('adversary')
        for epoch in range(0, args.adv_pretrain_epochs):
            # Train one epoch
            train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)

    
    
    print('Training:')
    optimizer.mode('train')
    for epoch in range(args.start_epoch, args.epochs):
        
        # Train one epoch
        train_one_epoch(train_loader, model, optimizer, criterion, epoch, args)
        
        # Evaluate performance on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        # TODO: Find a better metric for determining which checkpoint is "best"
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch+1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict()},
            is_best)


def train_one_epoch(train_loader, model, optimizer, criterion, epoch, args):
    
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
        
        # Adjust precision
        if args.precision == 'double':
            images = images.double()
        elif args.precision == 'half':
            images = images.half()

        # Take predictive step if requested
        pstep = 1.0 if args.prediction == optimizer.step_type() else 0.0
        with optimizer.lookahead(pstep):

            # Compute outputs and losses
            if args.use_gpu:
                target = target.cuda()
            (task_output,adv_output) = model(images)
            task_loss = criterion(task_output, target)
            adv_loss = criterion(adv_output, target)
            combined_loss = task_loss - args.lam*adv_loss
            
            # Choose relevant loss from step type and optimizer mode
            if optimizer.step_type() == 'adversary':
                loss = adv_loss
            elif optimizer.step_type() == 'task':
                loss = task_loss if optimizer._mode == 'task' else combined_loss
            
            # Measure accuracy and record losses
            acc1, acc5 = accuracy(task_output, target, topk=(1,5))
            task_losses.update(task_loss.item(), images.size(0))
            adv_losses.update(adv_loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
        # Compute gradients and take step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Automatically alternates between task and adversary
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % args.print_freq == 0:
            progress.display(i)
    

def validate(val_loader, model, criterion, args):
    
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
        for i, (images, target) in enumerate(val_loader):
            target = target.cuda()
            
            # Compute outputs
            task_output,adv_output = model(images)
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
            
            if i % args.print_freq == 0:
                progress.display(i)
            
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
    
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
    
    
   
# Command-line arg parser exposed for importing

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training - '
                    'With Texture Bias Adversary')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg11_adv', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + \
                    ' (default: vgg11_adv)')
parser.add_argument('-f', '--feature-layers', metavar='N', type=int, default=None,
                    help='number of shared feature layers')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrain-epochs', type=int, nargs='+', metavar=('PTE', 'adv_PTE'),
                    default=[10,10],
                    help='epochs to pretrain network branches separately [adversary '
                    'pretrain epochs]')
parser.add_argument('--adv-pretrain-epochs', type=int, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256) before dividing '
                    'batches among GPUs for data parallelism')
parser.add_argument('--lr', '--learning-rate', type=float, nargs='+', metavar=('LR', 'adv_LR'),
                    default=[0.1,0.1],
                    help='initial learning rate [adversary learning rate]')
parser.add_argument('--adv-lr', '--adversary-learning-rate', type=float, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument('--momentum', type=float, nargs='+', metavar=('M', 'adv_M'),
                    default=[0.0,0.0],
                    help='momentum [adversary momentum]')
parser.add_argument('--adv-momentum', '--adversary-momentum', type=float, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', nargs='?',
                    choices=['task','both','none'], const='both', default='none',
                    help='use pre-trained model [\'task\': use pretrained task model only]')
parser.add_argument('--eta', type=int, metavar='N', default=1,
                    help='adversary steps per task step (default: 1)')
parser.add_argument('--lambda', dest='lam', type=float, default=1.0, metavar='L',
                    help='relative weight of adversary loss wrt. task loss')
parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                    help='disable gpu use and data parallelism')
parser.add_argument('--prediction', dest='prediction', nargs='?',
                    choices=['task','adversary','none'], const='adversary', default='none',
                    help='enable gradient prediction on up to one branch')
parser.add_argument('--precision', dest='precision', default='float',
                    choices=['half', 'float', 'double'],
                    help='model precision')


if __name__ == '__main__':

    # Reformat args
    args = parser.parse_args()
    if args.adv_lr is None:
        args.adv_lr = args.lr[0] if len(args.lr) == 1 else args.lr[1]
    args.lr = args.lr[0]
    if args.adv_momentum is None:
        args.adv_momentum = args.momentum[0] if len(args.momentum) == 1 else args.momentum[1]
    args.momentum = args.momentum[0]
    if args.adv_pretrain_epochs is None:
        args.adv_pretrain_epochs = args.pretrain_epochs[0] if len(args.pretrain_epochs) == 1 \
                else args.pretrain_epochs[1]
    args.pretrain_epochs = args.pretrain_epochs[0]
    if args.prediction != "none" and args.eta != 1:
        warnings.warn('Eta must be 1 when using prediction ({} found). Proceeding with eta=1')
        args.eta = 1

    main(args)

    
    
    
    





