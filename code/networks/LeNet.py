import os
import sys
sys.path.extend([os.path.join(os.path.expanduser('~'),'pytorch_library')])

import torch
if torch.__version__!='1.0.0':
	raise Exception('Pytorch version must be 1.0.0')
import torch.nn as nn

class LeNet():
	pass
'''
from pytorch_library import apply_conv, apply_linear,return_activation,apply_pool

class LeNet(nn.Module):
	def __init__(self,database):
		super(LeNet,self).__init__()
		if database=='cifar10':
			input_kernels,reshape=3,4
		elif database=='mnist':
			input_kernels,reshape=1,3
		else:
			raise Exception('not implemented')

		conv1=apply_conv(input_kernels,32,3,'relu',drop=0.2,bn=True)
		conv2=apply_conv(32,64,3,'relu',drop=0.3,bn=True)
		conv3=apply_conv(64,128,3,'relu',drop=0.4,bn=True)
		mp=apply_pool(2)

		self.reshape=128*reshape*reshape

		fc1=apply_linear(self.reshape,1024,'relu',drop=0.5,bn=True)
		fc2=apply_linear(1024,1024,'relu',drop=0.5,bn=True)
		fc3=apply_linear(1024,10,'linear',drop=0.0,bn=True)

		self.forward_conv=nn.Sequential(conv1,mp,conv2,mp,conv3,mp)
		self.forward_fc=nn.Sequential(fc1,fc2,fc3)

	def forward(self,x):
		self.train()
		return self.forward_fc(self.forward_conv(x).view(-1,self.reshape))
		
	def cost(self,o,t):
		return nn.functional.cross_entropy(o,t)

	def forward_test(self,x):
		self.eval()
		return self.forward_fc(self.forward_conv(x).view(-1,self.reshape))		

	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()
'''
