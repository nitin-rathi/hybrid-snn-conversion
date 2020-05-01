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
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def find_threshold(batch_size=512, timesteps=2500):
    
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
    model.module.network_update(timesteps=timesteps, leak=1.0)

    pos=0
    thresholds=[]
    
    def find(layer, pos):
        max_act=0
        
        if layer == (len(model.module.features) + len(model.module.classifier) -1):
            return None

        f.write('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()
                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    pos = pos+1
                    f.write(' {}'.format(thresholds))
                    model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
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
    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    global learning_rate

    model.module.network_update(timesteps=timesteps, leak=leak)
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))
    
    #total_loss = 0.0
    #total_correct = 0
    model.train()
       
    #current_time = start_time
    #model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(train_loader):
               
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) 
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()        
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(),data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
                
        if (batch_idx+1) % 100000 == 0:
            temp1 = []
            for value in model.module.threshold.values():
                temp1 = temp1+[round(value.item(),2)]
            f.write('\nEpoch: {} batch: {} train_loss: {:.4f} train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        model.module.leak.item(),
                        model.module.timesteps
                        )
                    )
    f.write('\nEpoch: {} LR: {:.1e}, train_loss: {:.4f} train_acc: {:.4f}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        )
                    )
      
def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        if epoch>5 and max_accuracy<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        print_accuracy_every_batch = False
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if print_accuracy_every_batch:
                
                f.write('\nAccuracy: {}/{}({:.4f})'
                    .format(
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
        
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]    
        
        if top1.avg>max_accuracy:
            max_accuracy = top1.avg
             
            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : leak,
                    'activation'            : activation
                }
            try:
                os.mkdir('./trained_models/snn/')
            except OSError:
                pass 
            filename = './trained_models/snn/'+identifier+'.pth'
            torch.save(state,filename)    
        
            #if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        f.write(' test_loss: {:.4f}, current: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.40 0.65 0.85',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=100,                type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.7,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    

    args = parser.parse_args()

    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
           
    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    dropout             = args.dropout
    kernel_size         = args.kernel_size

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/snn/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)
    log_file+=identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file



    f.write('\n Run on time: {}'.format(datetime.datetime.now()))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    
    # Training settings
    
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    
    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
    
    elif dataset == 'MNIST':
        trainset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        testset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

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
        model = VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    
    elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN_STDB(name = architecture, labels=labels, timesteps=timesteps,leak_mem=leak_mem, STDP_alpha=STDP_alpha, STDP_beta=STDP_beta)

    # if freeze_conv:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model) 
    
    if pretrained_ann:
        
        if architecture[0:3].lower() == 'vgg':
            state = torch.load(pretrained_ann, map_location='cpu')
            cur_dict = model.state_dict()     
            for key in state['state_dict'].keys():
                if key in cur_dict:
                    if (state['state_dict'][key].shape == cur_dict[key].shape):
                        cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                        f.write('\n Loaded {} from {}'.format(key, pretrained_ann))
                    else:
                        f.write('\n Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
                else:
                    f.write('\n Loaded weight {} not present in current model'.format(key))
            model.load_state_dict(cur_dict)
            
            #If thresholds present in loaded ANN file
            if 'thresholds' in state.keys():
                thresholds = state['thresholds']
                f.write('\n Thresholds loaded from trained ANN: {}'.format(thresholds))
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])

            else:
                thresholds = find_threshold(batch_size=512, timesteps=1000)
                model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
                
                #Save the threhsolds in the ANN file
                temp = {}
                for key,value in state.items():
                    temp[key] = value
                temp['thresholds'] = thresholds
                torch.save(temp, pretrained_ann)
   
        elif architecture[0:3].lower() == 'res':
            state = torch.load(pretrained_state, map_location='cpu')
            f.write('\n Variables loaded from pretrained model:')
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    f.write('\n {} : {}'.format(key, value))
                else:
                    f.write('\n {}: '.format(key))
            model.load_state_dict(state['state_dict'])
    
    if pretrained_snn:
        
        if architecture[0:3].lower() == 'vgg':
            state = torch.load(pretrained_snn, map_location='cpu')
            cur_dict = model.state_dict()     
            for key in state['state_dict'].keys():
                if key in cur_dict:
                    if (state['state_dict'][key].shape == cur_dict[key].shape):
                        cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                        f.write('\n Loaded {} from {}'.format(key, pretrained_snn))
                    else:
                        f.write('\n Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
                else:
                    f.write('\n Loaded weight {} not present in current model'.format(state['state_dict'][key]))
            model.load_state_dict(cur_dict)

            if 'thresholds' in state.keys():
                if state['timesteps']!=timesteps or state['leak']!=leak:
                    f.write('\n Timesteps/Leak mismatch between loaded SNN and current simulation timesteps/leak, current timesteps/leak {}/{}, loaded timesteps/leak {}/{}'.format(timesteps, leak, state['timesteps'], state['leak']))
                thresholds = state['thresholds']
                model.module.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])
            else:
                f.write('\n Loaded SNN model does not have thresholds')

    #model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
    max_accuracy = 0
    
    #print(model)
    #f.write('\n Threshold: {}'.format(model.module.threshold))

    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))




