#---------------------------------------------------
# Imports
#---------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from self_models import *
import sys
import os
import shutil

use_cuda = True
torch.manual_seed(0)
if torch.cuda.is_available() and use_cuda:
    print ("\n \t ------- Running on GPU -------")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def find_threshold(ann_thresholds, loader):
    
    pos=0
    
    def find(layer, pos):
        max_act=0
        
        if architecture.lower().startswith('vgg'):
            if layer == (len(model.module.features) + len(model.module.classifier) -1):
                return None

        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                model.module.network_init(timesteps)
                output = model(data, 0, find_max_mem=True, max_mem_layer=layer)
                
                if output>max_act:
                    max_act = output.item()
                f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    ann_thresholds[pos] = max_act
                    pos = pos+1
                
                    model.module.threshold_init(scaling_threshold=scaling_threshold, reset_threshold=reset_threshold, thresholds = ann_thresholds[:], default_threshold=default_threshold)
                    break
        return pos

    if architecture.lower().startswith('vgg'):              
        for l in model.module.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
        
        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                pos = find(int(l[0])+int(c[0])+1, pos)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)

def train(epoch, loader):

    global learning_rate, start_time
    learning_rate_use = learning_rate * (lr_decay_factor**(epoch//lr_adjust_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate_use
    
    f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))
    
    total_loss = 0.0
    total_correct = 0
    model.train()
       
    current_time = start_time
    model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(loader):
               
        if torch.cuda.is_available() and use_cuda:
            data, target = data.cuda(), target.cuda()
        
        t=0
        mem = 0
        spike =0
        mask = 0
        spike_count = 0
               
        optimizer.zero_grad()
        while t<timesteps:
            
            output, mem, spike, mask, spike_count = model(data, t, mem, spike, mask, spike_count) 
            output = output/(t+update_interval)
            loss = criterion(output, target)
            loss = F.cross_entropy(output,target)
            loss.backward()
            t = t + update_interval
            total_loss = loss.item()
       
        optimizer.step()        
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        total_correct += correct.item()
        
        if (batch_idx+1) % 10 == 0:
            
            f.write('\nEpoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Current:[{}/{} ({:.2f}%)] Total:[{}/{} ({:.2f}%)] Time: {}({})'.format(
                epoch,
                (batch_idx+1) * len(data),
                len(loader.dataset),
                100. * (batch_idx+1) / len(loader),
                total_loss/(batch_idx+1),
                correct.item(),
                data.size(0),
                100. * correct.item()/data.size(0),
                total_correct,
                data.size(0)*(batch_idx+1),
                100. * total_correct/(data.size(0)*(batch_idx+1)),
                datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds),
                datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
                )
            )
            current_time = datetime.datetime.now()
      
def test(epoch, loader):

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        is_best = False
        print_accuracy_every_batch = True
        global max_correct
        
        for batch_idx, (data, target) in enumerate(loader):
                        
            if torch.cuda.is_available() and use_cuda:
                data, target = data.cuda(), target.cuda()
            
            model.module.network_init(timesteps)
            output, _, _, _, spike_count = model(data, 0)
            
            #for key in spike_count.keys():
            #    print('Key: {}, Average: {:.3f}'.format(key, (spike_count[key].sum()/spike_count[key].numel())))
            
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            if print_accuracy_every_batch:
                
                f.write('\nAccuracy: {}/{}({:.2f}%)'.format(
                    correct.item(),
                    (batch_idx+1)*data.size(0),
                    100. * correct.item() / ((batch_idx+1)*data.size(0))
                    )
                )
            
        if correct>max_correct:
            max_correct = correct
            is_best = True          
        
        state = {
                'accuracy'              : max_correct.item()/len(test_loader.dataset),
                'epoch'                 : epoch,
                'state_dict'            : model.state_dict(),
                'optimizer'             : optimizer.state_dict(),
                'thresholds'            : ann_thresholds,
                'timesteps'             : timesteps,
                'leak_mem'              : leak_mem,
                'scaling_threshold'     : scaling_threshold,
                'activation'            : activation
            }
        filename = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+str(update_interval)+'.pth'
        #torch.save(state,filename)    
        
        #if is_best:
        #    shutil.copyfile(filename, 'best_'+filename)

        f.write('\nTest set: Loss: {:.6f}, Current: {:.2f}%, Best: {:.2f}%\n'.  format(
            total_loss/(batch_idx+1), 
            100. * correct.item() / len(test_loader.dataset),
            100. * max_correct.item() / len(test_loader.dataset)
            )
        )
   
dataset             = 'CIFAR10' # {'CIFAR10', 'CIFAR100', 'IMAGENET'}
batch_size          = 64
timesteps           = 100
update_interval     = timesteps
num_workers         = 4
leak_mem            = 1.0 #{0.99, 1.0}
scaling_threshold   = 0.7
reset_threshold     = 0.0
default_threshold   = 1.0
activation          = 'Linear' # {'Linear', 'STDB'}
architecture        = 'VGG16' # {'VGG5','VGG9','VGG16','RESNET18','RESNET34'}
print_to_file       = False
log_file            = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+str(update_interval)+'_'+activation.lower()+'.log'
pretrained          = True
pretrained_state    = '/home/min/a/rathi2/SNN_backprop/trained_models/CIFAR10/SNN/snn_vgg16_cifar10.pth'
find_thesholds      = False
freeze_conv         = False
resume              = False
#resume              = './snn_vgg11_cifar100_125.pth'
learning_rate       = 1e-4
lr_adjust_interval  = 5
lr_decay_factor     = 0.5 # {0.1, 0.5, 1.0}
STDP_alpha          = 0.3
STDP_beta           = 0.01

if print_to_file:
    f = open(log_file, 'w', buffering=1)
else:
    f = sys.stdout

normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     normalize])
transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

if dataset == 'CIFAR10':
    trainset    = datasets.CIFAR10(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
    testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=                               transform_test)
    labels      = 10

elif dataset == 'CIFAR100':
    trainset    = datasets.CIFAR100(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
    testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=                               transform_test)
    labels      = 100

elif dataset == 'IMAGENET':
    labels      = 1000
    traindir    = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'train')
    valdir      = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'val')
    trainset    = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    testset     = datasets.ImageFolder(
                        valdir,
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])) 


train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)

if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak_mem=leak_mem, STDP_alpha=STDP_alpha, STDP_beta=STDP_beta)
    
elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN_STDB(name = architecture, labels=labels, timesteps=timesteps,leak_mem=leak_mem, STDP_alpha=STDP_alpha, STDP_beta=STDP_beta)

if freeze_conv:
    for param in model.features.parameters():
        param.requires_grad = False
#Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
model = nn.DataParallel(model) 
if pretrained:
        
    if architecture[0:3].lower() == 'vgg':
        state = torch.load(pretrained_state, map_location='cpu')
        f.write('\n Variables loaded from pretrained model:')
        
        for key, value in state.items():
            if isinstance(value, (int, float)):
                f.write('\n {} : {}'.format(key, value))
            else:
                f.write('\n {}: '.format(key))
        
        model.load_state_dict(state['state_dict'])

   
    elif architecture[0:3].lower() == 'res':
        state = torch.load(pretrained_state, map_location='cpu')
        f.write('\n Variables loaded from pretrained model:')
        for key, value in state.items():
            if isinstance(value, (int, float)):
                f.write('\n {} : {}'.format(key, value))
            else:
                f.write('\n {}: '.format(key))
        model.load_state_dict(state['state_dict'])
    
#model = nn.DataParallel(model) 
if torch.cuda.is_available() and use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4, amsgrad=False)
criterion = nn.CrossEntropyLoss()
max_correct = 0
start_epoch = 1

f.write('\nDataset                  :{} '.format(dataset))
f.write('\nBatch Size               :{} '.format(batch_size))
f.write('\nTimesteps                :{} '.format(timesteps))
f.write('\nUpdate Interval (time)   :{} '.format(update_interval))
f.write('\nMembrane Leak            :{} '.format(leak_mem))
f.write('\nScaling Threshold        :{} '.format(scaling_threshold))
f.write('\nActivation               :{} '.format(activation))
f.write('\nArchitecture             :{} '.format(architecture))
if pretrained:
    f.write('\nPretrained Weight File   :{} '.format(pretrained_state))
elif resume:
    f.write('\nResumed from state       :{} '.format(resume))
f.write('\nStarting Learning Rate   :{} '.format(learning_rate))
f.write('\nLR Adjust Interval       :{} '.format(lr_adjust_interval))
f.write('\nLR Decay Factor          :{} '.format(lr_decay_factor))
f.write('\nSTDP_alpha               :{} '.format(STDP_alpha))
f.write('\nSTDP_beta                :{} '.format(STDP_beta))
f.write('\nOptimizer                :{} '.format(optimizer))
f.write('\nCriterion                :{} '.format(criterion))
f.write('\n{}'.format(model))

start_time = datetime.datetime.now()

ann_thresholds = []

if architecture.lower().startswith('vgg'):
    for l in model.module.features.named_children():
    
        if isinstance(l[1], nn.Conv2d):
            ann_thresholds.append(default_threshold)
    
    for l in model.module.classifier.named_children():
    
        if isinstance(l[1], nn.Linear):
            ann_thresholds.append(default_threshold)
    
if architecture.lower().startswith('res'):
    for l in model.module.pre_process.named_children():
        if isinstance(l[1], nn.Conv2d):
            ann_thresholds.append(default_threshold)

#VGG16 Imagenet thresholds
#ann_thresholds = [10.16, 11.49, 2.65, 2.30, 0.77, 2.75, 1.33, 0.67, 1.13, 1.12, 0.43, 0.73, 1.08, 0.16, 0.58]

#ResNet34 Imagenet thresholds
#ann_thresholds = [8.73, 4.88, 5.80]

#VGG11 CIFAR100 thresholds
#ann_thresholds = [15.99, 1.85, 2.04, 1.33, 0.34, 0.83, 0.87, 0.31, 1.73, 1.40, 1.41]

#VGG16 CIFAR10 thresholds
ann_thresholds = [4.26, 2.87, 0.48, 1.34, 0.21, 0.87, 1.12, 0.17, 1.30, 4.46, 0.56, 2.18, 1.70, 1.35, 1.33]

#VGG5 CIFAR10 thresholds
#ann_thresholds = [12.70, 1.39, 1.65, 0.17, 0.54]


#Uncomment to find firing thresholds
#if pretrained and find_thesholds:
#    find_threshold(ann_thresholds, train_loader)
    
thresholds_set = model.module.threshold_init(scaling_threshold=scaling_threshold, reset_threshold=reset_threshold, thresholds = ann_thresholds[:], default_threshold=default_threshold)
f.write('\n Threshold: {}'.format(thresholds_set))

for epoch in range(start_epoch, 50):
    
    #train(epoch, train_loader)
    test(epoch, test_loader)    

f.write('\nHighest accuracy: {:.2f}%'.format(100*max_correct.item()/len(test_loader.dataset)))


