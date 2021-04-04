import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
import copy

cfg = {
	'resnet6'	: [1,1,0,0],
	'resnet12' 	: [1,1,1,1],
	'resnet20'	: [2,2,2,2],
	'resnet34'	: [3,4,6,3]
}

class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*0.9).float(),torch.sign(input))
		return out

class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike):
        
		ctx.save_for_backward(last_spike)
		out = torch.zeros_like(input).cuda()
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		return grad*grad_input, None

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout):
        #print('In __init__ BasicBlock')
        #super(BasicBlock, self).__init__()
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        self.identity = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.identity = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, dic):
        #print('In forward BasicBlock')
        #pdb.set_trace()
        out_prev 		= dic['out_prev']
        pos 			= dic['pos']
        act_func 		= dic['act_func']
        mem 			= dic['mem']
        spike 			= dic['spike']
        mask 			= dic['mask']
        threshold 		= dic['threshold']
        t 				= dic['t']
        leak			= dic['leak']
        #find_max_mem 	= dic['find_max_mem']
        inp				= out_prev.clone()
        # for m in mem:
        # 	m.detach_()
        # for s in spike:
        # 	s.detach_()

        mem_thr 	= (mem[pos]/threshold[pos]) - 1.0
        out 		= act_func(mem_thr, (t-1-spike[pos]))
        rst 		= threshold[pos] * (mem_thr>0).float()
        spike[pos] 	= spike[pos].masked_fill(out.bool(),t-1)
        mem[pos] 		= leak*mem[pos] + self.residual[0](inp) - rst
        out_prev  	= out.clone()

        out_prev 	= out_prev * mask[pos]

        mem_thr 	= (mem[pos+1]/threshold[pos+1]) - 1.0
        out 		= act_func(mem_thr, (t-1-spike[pos+1]))
        rst 		= threshold[pos+1] * (mem_thr>0).float()
        spike[pos+1] 	= spike[pos+1].masked_fill(out.bool(),t-1)
        
        #if find_max_mem:
        #	return (self.delay_path[2](out_prev) + self.shortcut(inp)).max()
        #if t==199:
        #	print((self.delay_path[3](out_prev) + self.shortcut(inp)).max())
        #if len(self.shortcut)>0:
        
        mem[pos+1] 		= leak*mem[pos+1] + self.residual[3](out_prev) + self.identity(inp) - rst
        #else:
        #	mem[1] 		= leak_mem*mem[1] + self.delay_path[1](out_prev) + inp - rst
        
        out_prev  	= out.clone()
        
        #result				= {}
        #result['out_prev'] 	= out.clone()
        #result['mem'] 		= mem[:]
        #result['spike'] 	= spike[:]
        
        #pdb.set_trace()
        return out_prev

class RESNET_SNN_STDB(nn.Module):
	
	#all_layers = []
	#drop 		= 0.2
	def __init__(self, resnet_name, activation='Linear', labels=10, timesteps=75, leak=1.0, default_threshold=1.0, alpha=0.5, beta=0.035, dropout=0.2, dataset='CIFAR10'):

		super().__init__()
		
		self.resnet_name	= resnet_name.lower()
		if activation == 'Linear':
			self.act_func 	= LinearSpike.apply
		elif activation == 'STDB':
			self.act_func	= STDB.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		self.leak 	 		= torch.tensor(leak)
		STDB.alpha 			= alpha
		STDB.beta 			= beta 
		self.dropout 		= dropout
		self.dataset 		= dataset
		self.input_layer 	= PoissonGenerator()
		self.threshold 		= {}
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}

		self.pre_process    = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.AvgPool2d(2)
                                )
		block 				= BasicBlock
		self.in_planes      = 64
		
		self.layer1 		= self._make_layer(block, 64, cfg[self.resnet_name][0], stride=1, dropout=self.dropout)
		self.layer2 		= self._make_layer(block, 128, cfg[self.resnet_name][1], stride=2, dropout=self.dropout)
		self.layer3 		= self._make_layer(block, 256, cfg[self.resnet_name][2], stride=2, dropout=self.dropout)
		self.layer4 		= self._make_layer(block, 512, cfg[self.resnet_name][3], stride=2, dropout=self.dropout)
		#self.avgpool 		= nn.AvgPool2d(2)
		self.classifier     = nn.Sequential(
									nn.Linear(512*2*2, labels, bias=False)
									)

		self.layers = {1: self.layer1, 2: self.layer2, 3: self.layer3, 4:self.layer4}

		self._initialize_weights2()
		
		for l in range(len(self.pre_process)):
			if isinstance(self.pre_process[l],nn.Conv2d):
				self.threshold[l] = torch.tensor(default_threshold)

		pos = len(self.pre_process)
				
		for i in range(1,5):

			layer = self.layers[i]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						self.threshold[pos] = torch.tensor(default_threshold)
						pos=pos+1
				
	def _initialize_weights2(self):

		for m in self.modules():
			
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):
    	
		self.scaling_factor = scaling_factor

		for pos in range(len(self.pre_process)):
			if isinstance(self.pre_process[pos],nn.Conv2d):
				if thresholds:
					self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)

	def _make_layer(self, block, planes, num_blocks, stride, dropout):

		if num_blocks==0:
			return nn.Sequential()
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, dropout))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
		self.leak 		= torch.tensor(leak)
	
	def neuron_init(self, x):
		
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)

		self.mem 	= {}
		self.spike 	= {}
		self.mask 	= {}

		#self.width 		= [x.size(2), x.size(2)//3, x.size(2)//6, 19, 10]
		#self.height 	= [x.size(3), x.size(3)//3, x.size(3)//6, 19, 10]

		# Pre process layers
		for l in range(len(self.pre_process)):
			
			if isinstance(self.pre_process[l], nn.Conv2d):
				self.mem[l] = torch.zeros(self.batch_size, self.pre_process[l].out_channels, self.width, self.height)

			elif isinstance(self.pre_process[l], nn.Dropout):
				self.mask[l] = self.pre_process[l](torch.ones(self.mem[l-2].shape))
			elif isinstance(self.pre_process[l], nn.AvgPool2d):
				
				self.width 	= self.width//self.pre_process[l].kernel_size
				self.height = self.height//self.pre_process[l].kernel_size 

		pos = len(self.pre_process)
		for i in range(1,5):
			layer = self.layers[i]
			self.width = self.width//layer[0].residual[0].stride[0]
			self.height = self.height//layer[0].residual[0].stride[0]
			for index in range(len(layer)):
				for l in range(len(layer[index].residual)):
					if isinstance(layer[index].residual[l],nn.Conv2d):
						self.mem[pos] = torch.zeros(self.batch_size, layer[index].residual[l].out_channels, self.width, self.height)
						pos = pos + 1
					elif isinstance(layer[index].residual[l],nn.Dropout):
						self.mask[pos-1] = layer[index].residual[l](torch.ones(self.mem[pos-1].shape))
		
		#average pooling before final layer
		#self.width 	= self.width//self.avgpool.kernel_size
		#self.height = self.height//self.avgpool.kernel_size

		#final classifier layer
		self.mem[pos] = torch.zeros(self.batch_size, self.classifier[0].out_features)

		self.spike = copy.deepcopy(self.mem)
		for key, values in self.spike.items():
			for value in values:
				value.fill_(-1000)

	def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
		self.neuron_init(x)
		
		# for key, values in self.mem.items():
		# 	values[0].detach_()
		# for key, values in self.spike.items():
		# 	values[0].detach_()
		# for key, values in self.mask.items():
		# 	values.detach_()

		max_mem = 0.0
		
		for t in range(self.timesteps):

			out_prev = self.input_layer(x)
					
			for l in range(len(self.pre_process)):
							
				if isinstance(self.pre_process[l], nn.Conv2d):
					
					if find_max_mem and l==max_mem_layer:
						if (self.pre_process[l](out_prev)).max()>max_mem:
							max_mem = (self.pre_process[l](out_prev)).max()
						break
					
					mem_thr 		= (self.mem[l]/self.threshold[l]) - 1.0
					out 			= self.act_func(mem_thr, (t-1-self.spike[l]))
					rst 			= self.threshold[l] * (mem_thr>0).float()
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
					
					# print('mem:{} l:{}'.format(self.mem[l][0].shape, l))
					# print('pre:{} l:{}'.format((self.pre_process[l](out_prev)).shape,l))
					# print('rst:{} l:{}'.format(rst.shape,l))
					self.mem[l] 	= self.leak*self.mem[l] + self.pre_process[l](out_prev) - rst
					out_prev  			= out.clone()

				elif isinstance(self.pre_process[l], nn.AvgPool2d):
					out_prev 		= self.pre_process[l](out_prev)
				
				elif isinstance(self.pre_process[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]
			
			if find_max_mem and max_mem_layer<len(self.pre_process):
				continue
				
			pos 	= len(self.pre_process)
			
			for i in range(1,5):
				layer = self.layers[i]
				for index in range(len(layer)):
					out_prev = layer[index]({'out_prev':out_prev.clone(), 'pos': pos, 'act_func': self.act_func, 'mem':self.mem, 'spike':self.spike, 'mask':self.mask, 'threshold':self.threshold, 't': t, 'leak':self.leak})
					pos = pos+2
			
			#out_prev = self.avgpool(out_prev)
			out_prev = out_prev.view(self.batch_size, -1)

			# Compute the classification layer outputs
			self.mem[pos] = self.mem[pos] + self.classifier[0](out_prev)
			
		if find_max_mem:
			return max_mem
				
		return self.mem[pos]	
