#############################
#   @author: Nitin Rathi    #
#############################
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
from self_models import *

def train(epoch, loader):

    global learning_rate
            
    if epoch in [125, 200, 250]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
            learning_rate = param_group['lr']
    
    total_correct   = 0
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        
        start_time = datetime.datetime.now()

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
                
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()
        #exit(0)
        loss.backward()
        optimizer.step()
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        total_correct += correct.item()

        # if (batch_idx+1) % 100 == 0:
        #     #f.write('\nconv1: {:.2e}, conv2: {:.2e}, conv3: {:.2e}'.format(
        #     #model.conv1.weight.mean().item(),
        #     #model.conv2.weight.mean().item(),
        #     #model.conv3.weight.mean().item(),
        #     #))
            
        #     f.write('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Current:[{}/{} ({:.2f}%)] Total:[{}/{} ({:.2f}%)] Time: {}'.format(
        #         epoch,
        #         (batch_idx+1) * len(data),
        #         len(loader.dataset),
        #         100. * (batch_idx+1) / len(loader),
        #         loss,
        #         correct.item(),
        #         data.size(0),
        #         100. * correct.item()/data.size(0),
        #         total_correct,
        #         data.size(0)*(batch_idx+1),
        #         100. * total_correct/(data.size(0)*(batch_idx+1)),
        #         datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        #         )
        #     )
    f.write('\n Epoch: {}, LR: {}, Train Loss: {:.6f}, Train accuracy: {:.4f}'.format(
            epoch,
            learning_rate,
            loss,
            total_correct/len(loader.dataset)
            )
        )

def test(loader):

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        global max_correct, start_time

        for batch_idx, (data, target) in enumerate(loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
        if correct>max_correct:
            max_correct = correct
            state = {
                    'accuracy'      : max_correct.item()/len(loader.dataset),
                    'epoch'         : epoch,
                    'state_dict'    : model.state_dict(),
                    'optimizer'     : optimizer.state_dict()
            }

            filename = 'ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
            torch.save(state,filename)
            
        f.write(' Test Loss: {:.6f}, Current: {:.2f}%, Best: {:.2f}%, Time: {}'.  format(
            total_loss/(batch_idx+1), 
            100. * correct.item() / len(loader.dataset),
            100. * max_correct.item() / len(loader.dataset),
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )
        # f.write('\n Time: {}'.format(
        #     datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
        #     )
        # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu',                    default=True,         type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,            type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',    type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,           type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',      type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16',' VGG19'])
    parser.add_argument('-lr','--learning_rate',    default=1e-2,         type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_CIFAR10',     default='',           type=str,       help='pretrained CIFAR10 model to initialize CIFAR100 training')
    parser.add_argument('--log',                    action='store_true',                 help='to print the output on terminal or to log file')
    
    args=parser.parse_args()
    
    
    if args.log:
        log_file = 'ann_'+args.architecture.lower()+'_'+args.dataset.lower()+'.log'
        f= open(log_file, 'w', buffering=1)
    else:
        f=sys.stdout
    f.write('\n Log file for \'{}\', run on {}'.format(sys.argv[0],datetime.datetime.now()))
    f.write('\n\n')
    f.write('\n Dataset:{}'.format(args.dataset))
    f.write('\n Batch size: {}'.format(args.batch_size))
    f.write('\n Architecture: {}'.format(args.architecture))
    
    # Training settings
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu:
        f.write("\n \t ------- Running on GPU -------")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    dataset         = args.dataset
    batch_size      = args.batch_size
    
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    if dataset == 'CIFAR100':
        train_dataset   = datasets.CIFAR100(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        train_dataset   = datasets.CIFAR10(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'MNIST':
        train_dataset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        test_dataset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                               shuffle=False)
    
    architecture = args.architecture
    model = VGG(architecture, labels)
    f.write('\n{}'.format(model))
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    
    if dataset == 'CIFAR100' and args.pretrained_CIFAR10:
        state=torch.load(args.pretrained_CIFAR10)
        if 'classifier.6.weight' in state['state_dict']:
            state['state_dict'].pop('classifier.6.weight')
            state['state_dict']['classifier.6.weight'] = model.state_dict()['classifier.6.weight']
        elif 'module.classifier.6.weight' in state['state_dict']:
            state['state_dict'].pop('module.classifier.6.weight')
            state['state_dict']['module.classifier.6.weight'] = model.state_dict()['module.classifier.6.weight']
    
        model.load_state_dict(state['state_dict'])
    
    model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    
    learning_rate = args.learning_rate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    max_correct = 0
    
    for epoch in range(1, 300):    
        start_time = datetime.datetime.now()
        train(epoch, train_loader)
        test(test_loader)
           
    f.write('Highest accuracy: {:.2f}%'.format(100*max_correct.item()/len(test_loader.dataset)))
