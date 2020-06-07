import main
import models
import mosaic_mnist

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import shutil
import numpy as np
import argparse

def init(subfolder=''):
    os.makedirs(os.path.join('trained_models', subfolder), exist_ok=True)


######################## EXPERIMENT 1 ########################
# Train a regular model on consistent Mosaic MNIST, train an #
# adversarial model based on it, then compare the two's      #
# performances on all three versions of Mosaic MNIST         #
##############################################################

def experiment1(**kwargs):
    init('exp1+bg')
    log_path = 'trained_models/exp1+bg'
    ckpt_path = log_path + '/checkpoint.pth.tar'
    vgg_path = log_path + '/vgg13.pth.tar'
    vgg_adv_path = log_path + '/vgg13_adv.pth.tar'
    
    BATCH_SIZE = 16
    
    # Train task model, pretrain adversary
    args = None
    if not os.path.exists(vgg_path):
        args = main.parser.parse_args(['dat',
                                       '--dataset',             'mosaic_mnist',
                                       '--arch',                'vgg13_adv',
                                       #'--no-gpu',
                                       '--precision',           'half',
                                       '--epochs',              '0',
                                       '--pretrain-epochs',     '2',
                                       '--adv-pretrain-epochs', '2',
                                       '--batch-size',          str(BATCH_SIZE),
                                       '--lr',                  '0.01',
                                       '--adv-lr',              '0.01', 
                                       '--momentum',            '0.',
                                       '--adv-momentum',        '0.',
                                       '--log-path',            log_path,
                                       #'--no-tensorboard'
                                       ])
        args = main.reformat_args(args)
        print('> Training VGG13 model')
        BATCH_SIZE = main.main_reduce_batch(args) # Pretrain, return biggest working batch size
        shutil.move(ckpt_path, vgg_path)
    else:
        print('Trained VGG13 model found! Skipping VGG13 training.')
        
    # Interpret args for training/testing
    args = main.parser.parse_args(['dat',
                                   '--dataset',             'mosaic_mnist',
                                   '--arch',                'vgg13_adv',
                                   #'--no-gpu',
                                   '--precision',           'half',
                                   '--epochs',              '10',
                                   '--pretrain-epochs',     '0',
                                   '--adv-pretrain-epochs', '0',
                                   '--batch-size',          str(BATCH_SIZE),
                                   '--lr',                  '0.001',
                                   '--adv-lr',              '0.01', 
                                   '--momentum',            '0.',
                                   '--adv-momentum',        '0.',
                                   '--log-path',            log_path,
                                   '--load-from',           vgg_path,
                                   #'--no-tensorboard'
                                   ])
    args = main.reformat_args(args)
    for kw in kwargs: # Manual arg setting from command line
        kwargs[kw] = type(getattr(args, kw))(kwargs[kw]) # Convert to appropriate type
        setattr(args, kw, kwargs[kw])
        
    # These might have changed
    log_path = args.log_path
    ckpt_path = log_path + '/checkpoint.pth.tar'
    vgg_adv_path = log_path + '/vgg13_adv.pth.tar'

    # Train adversarial model
    if not os.path.exists(vgg_adv_path):
        print('> Training adversarial VGG13 model')
        BATCH_SIZE = main.main_reduce_batch(args)   # Train, return batch size
        shutil.move(ckpt_path, vgg_adv_path)
    else:
        print('Trained Adversarial VGG13 model found! Skipping adversarial VGG13 training.')
        
    # Test both models on Mosaic MNIST variants
    if not os.path.exists(log_path + '/results.np'):
        # Load saved models
        vgg_save = torch.load(vgg_path)
        vgg_model = models.vgg13_adv()
        vgg_model.load_state_dict(vgg_save['state_dict'])
        vgg_model = torch.nn.DataParallel(vgg_model.cuda()).half()
        
        vgg_adv_save = torch.load(vgg_adv_path)
        vgg_adv_model = models.vgg13_adv()
        vgg_adv_model.load_state_dict(vgg_adv_save['state_dict'])
        vgg_adv_model = torch.nn.DataParallel(vgg_adv_model.cuda()).half()
        
        # Load test sets
        print('> Loading test sets')
        tf = transforms.Compose([
                transforms.ToTensor(),
                mosaic_mnist.grayscale2color])
        dataloaders = dict()
        for name in ['consistent', 'inconsistent', 'malicious']:
            dset = mosaic_mnist.MnistMosaicDataset(name+'_test+bg', tf, True)   #TODO: +bg
            dataloaders[name] = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Test models on sets
        results = {'vgg13': {}, 'vgg13_adv': {}}
        criterion = torch.nn.CrossEntropyLoss()
        args = main.parser.parse_args(['dat', '--precision', 'half'])
        #args = main.parser.parse_args(['dat'])
        for name in ['consistent', 'inconsistent', 'malicious']:
            print('Testing VGG13 model on ' + name + ' Mosaic MNIST...')
            results['vgg13'][name] = main.validate(dataloaders[name], vgg_model, criterion, args)
            
        for name in ['consistent', 'inconsistent', 'malicious']:
            print('Testing Adversarial VGG13 model on ' + name + ' Mosaic MNIST...')
            results['vgg13_adv'][name] = main.validate(dataloaders[name], vgg_adv_model, criterion, args)
        
        # Save results
        with open(log_path + '/results.np', 'wb') as f:
            np.save(f, results)
    
    else: # Results already exist
        print('Results file found! Skipping testing of networks.')
        with open(log_path + '/results.np', 'rb') as f:
            results = np.load(f).item()
    
    # Print results
    print('\n\nRESULTS:\n')
    print('%-11s| %-13s| %-13s| %-13s' % ('Model', 'Consistent', 'Inconsistent', 'Malicious'))
    print('-'*11 + ('+' + '-'*14)*3)
    print('%-11s| %-13.2f| %-13.2f| %-13.2f' % ('vgg13', results['vgg13']['consistent'], results['vgg13']['inconsistent'], results['vgg13']['malicious']))
    print('%-11s| %-13.2f| %-13.2f| %-13.2f' % ('vgg13_adv', results['vgg13_adv']['consistent'], results['vgg13_adv']['inconsistent'], results['vgg13_adv']['malicious']))
    
    
    
    
    
    
def experiment2(**kwargs):
    init('exp2')
    log_path = 'trained_models/exp2'
    ckpt_path = log_path + '/checkpoint.pth.tar'
    vgg_path = log_path + '/vgg13.pth.tar'
    vgg_adv_path = log_path + '/vgg13_adv.pth.tar'
    
    BATCH_SIZE = 16
    
    args = None
    if not os.path.exists(vgg_path):
        args = main.parser.parse_args(['/z/dat/ImageNet_2012',
                                       '--dataset',             'imagenet',
                                       '--arch',                'vgg13_adv',
                                       '--precision',           'half',
                                       '--pretrained',          'task',
                                       '--epochs',              '0',
                                       '--pretrain-epochs',     '0',
                                       '--adv-pretrain-epochs', '1',
                                       '--batch-size',          str(BATCH_SIZE),
                                       '--lr',                  '0.01',
                                       '--adv-lr',              '0.01', 
                                       '--momentum',            '0.',
                                       '--adv-momentum',        '0.',
                                       '--log-path',            log_path,
                                       ])
        args = main.reformat_args(args)
        print('> Training VGG13 model')
        BATCH_SIZE = main.main_reduce_batch(args) # Pretrain, return biggest working batch size
        shutil.move(ckpt_path, vgg_path)
    else:
        print('Trained VGG13 model found! Skipping VGG13 training.')

    # Interpret args for training/testing
    args = main.parser.parse_args(['/z/dat/ImageNet_2012',
                                   '--dataset',             'imagenet',
                                   '--arch',                'vgg13_adv',
                                   '--precision',           'half',
                                   '--epochs',              '10',
                                   '--pretrain-epochs',     '0',
                                   '--adv-pretrain-epochs', '0',
                                   '--batch-size',          str(BATCH_SIZE),
                                   '--lr',                  '0.001',
                                   '--adv-lr',              '0.01', 
                                   '--momentum',            '0.',
                                   '--adv-momentum',        '0.',
                                   '--log-path',            log_path,
                                   '--load-from',           vgg_path,
                                   ])
    args = main.reformat_args(args)
    for kw in kwargs: # Manual arg setting from command line
        kwargs[kw] = type(getattr(args, kw))(kwargs[kw]) # Convert to appropriate type
        setattr(args, kw, kwargs[kw])
        
    # These might have changed
    log_path = args.log_path
    ckpt_path = log_path + '/checkpoint.pth.tar'
    vgg_adv_path = log_path + '/vgg13_adv.pth.tar'

    # Train adversarial model
    if not os.path.exists(vgg_adv_path):
        print('> Training adversarial VGG13 model')
        BATCH_SIZE = main.main_reduce_batch(args)   # Train, return batch size
        shutil.move(ckpt_path, vgg_adv_path)
    else:
        print('Trained Adversarial VGG13 model found! Skipping adversarial VGG13 training.')
    
    # Test both models on Mosaic MNIST variants
    if not os.path.exists(log_path + '/results.np'):
        # Load saved models
        vgg_save = torch.load(vgg_path)
        vgg_model = models.vgg13_adv()
        vgg_model.load_state_dict(vgg_save['state_dict'])
        vgg_model = torch.nn.DataParallel(vgg_model.cuda()).half()
        
        vgg_adv_save = torch.load(vgg_adv_path)
        vgg_adv_model = models.vgg13_adv()
        vgg_adv_model.load_state_dict(vgg_adv_save['state_dict'])
        vgg_adv_model = torch.nn.DataParallel(vgg_adv_model.cuda()).half()
    
        dataloader = main.make_dataloader(args, training=False)
        acc1_vgg = main.validate(dataloader, vgg_model, torch.nn.CrossEntropyLoss(), args)
        acc1_vgg_adv = main.validate(dataloader, vgg_adv_model, torch.nn.CrossEntropyLoss(), args)
        
        # Save results
        with open(log_path + '/results.np', 'wb') as f:
            np.save(f, {'acc1_vgg': acc1_vgg, 'acc1_vgg_adv': acc1_vgg_adv})
        
    else:
        print('Results found! Skipping testing.')
        
        with open(log_path + '/results.np', 'wb') as f:
            results = np.load(f)
        acc1_vgg = results['acc1_vgg']
        acc1_vgg_adv = results['acc1_vgg_adv']
        
        
    print('Results:')
    print('VGG13 Acc@1:      {}%'.format(acc1_vgg*100.))
    print('VGG13 Adv. Acc@1: {}%'.format(acc1_vgg_adv*100.))
    



def experiment3(**kwargs):
    init('exp3')
    log_path = 'trained_models/exp3'
    ckpt_path = log_path + '/checkpoint.pth.tar'
    pretrained_path = log_path + '/pretrained.pth.tar'
    trained_path = log_path + '/trained.pth.tar'

    BATCH_SIZE = 16

    args = main.parser.parse_args(['dat',
                                   '--dataset', 'mosaic_mnist',
                                   '--arch', 'twolayer_adv',
                                   '--precision', 'half',
                                   '--pretrained', 'none',
                                   '--epochs', '0',
                                   '--pretrain-epochs', '5',
                                   '--adv-pretrain-epochs', '5',
                                   '--batch-size', str(BATCH_SIZE),
                                   '--lr', '0.001',
                                   '--adv-lr', '0.01',
                                   '--momentum', '0.',
                                   '--adv-momentum', '0.',
                                   '--log-path', log_path,
                                   ])
    for kw in kwargs:
        kwargs[kw] = type(getattr(args, kw))(kwargs[kw])
        setattr(args, kw, kwargs[kw])

    args = main.reformat_args(args)
    print('> Pretraining two-layer model')
    BATCH_SIZE = main.main_reduce_batch(args)
    shutil.move(ckpt_path, pretrained_path)
    
    args.load_path = pretrained_path
    args.epochs = 10
    args.pretrain_epochs = 0
    args.adv_pretrain_epochs = 0

    print('> Training two-layer model')
    BATCH_SIZE = main.main_reduce_batch(args)
    shutil.move(ckpt_path, trained_path)

    pretrained_save = torch.load(pretrained_path)
    pretrained_model = models.twolayer_adv()
    pretrained_model.load_state_dict(pretrained_save['state_dict'])
    pretrained_model = torch.nn.DataParallel(pretrained_model.cuda()).half()

    trained_save = torch.load(trained_path)
    trained_model = models.twolayer_adv()
    trained_model.load_state_dict(trained_save['state_dict'])
    trained_model = torch.nn.DataParallel(trained_model.cuda()).half()

    # Load test sets
    print('> Loading test sets')
    tf = transforms.Compose([
            transforms.ToTensor(),
            mosaic_mnist.grayscale2color])
    dataloaders = dict()
    for name in ['consistent', 'inconsistent', 'malicious']:
        dset = mosaic_mnist.MnistMosaicDataset(name+'_test', tf, True)
        dataloaders[name] = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Test models on sets
    results = {'pretrained': {}, 'trained': {}}
    criterion = torch.nn.CrossEntropyLoss()
    args = main.parser.parse_args(['dat', '--precision', 'half'])
    #args = main.parser.parse_args(['dat'])
    for name in ['consistent', 'inconsistent', 'malicious']:
        print('Testing pretrained model on ' + name + ' Mosaic MNIST...')
        results['pretrained'][name] = main.validate(dataloaders[name], pretrained_model, criterion, args)
        
    for name in ['consistent', 'inconsistent', 'malicious']:
        print('Testing trained model on ' + name + ' Mosaic MNIST...')
        results['trained'][name] = main.validate(dataloaders[name], trained_model, criterion, args)
    
    with open(log_path + '/results.np', 'wb') as f:
        np.save(f, results)

    print('RESULTS:')
    print('%10s  %-12s  %-12s  %-12s' % ('Model', 'Consistent', 'Inconsistent', 'Malicious'))
    for model in ['pretrained', 'trained']:
        print('%10s: %-12.2f  %-12.2f  %-12.2f' % (model, results[model]['consistent'], results[model]['inconsistent'], results[model]['malicious']))




parser = argparse.ArgumentParser(description='Paper Experiments for ARTB in CNNs')
parser.add_argument('exps', default='1', metavar='N')
parser.add_argument('kwargs', nargs='*', metavar='**kwargs')

if __name__ == '__main__':
    args = parser.parse_args()
    experiments = {'1': experiment1, '2': experiment2, '3': experiment3} #TODO: Add others as they come
    for exp in args.exps:
        print('+-----------------------+')
        print('| RUNNING: Experiment {} |'.format(exp))
        print('+-----------------------+')
        kwargs = dict(zip(args.kwargs[0::2], args.kwargs[1::2]))
        experiments[exp](**kwargs)
    
    
