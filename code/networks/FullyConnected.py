import torch
if torch.__version__!='1.0.0':
	raise Exception('Pytorch version must be 1.0.0')
import torch.nn as nn

import os
import sys
sys.path.extend([os.path.expanduser('~/py3.7_PYTORCH/')])
#from pytorch_library import apply_conv, apply_linear,return_activation,apply_pool

class fc:
	pass
'''

class fc(nn.Module):
	def __init__(self):
		super(fc,self).__init__()
		self.dim=28*28
		fc1=apply_linear(self.dim,1024,'relu',drop=0.5,bn=True)
		fc2=apply_linear(1024,1024,'relu',drop=0.5,bn=True)
		fc3=apply_linear(1024,10,'linear',drop=0.0,bn=True)

		self.forward_fc=nn.Sequential(fc1,fc2,fc3)

	def forward(self,x):
		self.train()
		x=x.view(-1,self.dim)
		return self.forward_fc(x)
		
	def cost(self,o,t):
		return nn.functional.cross_entropy(o,t)

	def forward_test(self,x):
		self.eval()
		x=x.view(-1,self.dim)
		return self.forward_fc(x)		

	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()
'''
