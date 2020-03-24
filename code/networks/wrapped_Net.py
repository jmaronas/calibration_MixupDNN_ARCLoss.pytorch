# -*- coding: utf-8 -*-
## Author: Juan Maroñas PRHLT Research Center (jmaronas@prhlt.upv.es)

import torch
if torch.__version__!='1.0.0':
        raise Exception('Pytorch version must be 1.0.0')
'''DenseNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class wrappedNET(nn.Module):
	def __init__(self, wrapped_net):
		super(wrappedNET, self).__init__()
		self.model = wrapped_net

	def forward(self,x):
		return self.model.forward(x)

	def forward_test(self,x):
		return self.model.forward_test(x)

	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()


	def classification_acc(self,out,t):
		return (out.argmax(dim=1)==t).sum()

	def cost(self,o,t):
		return nn.functional.cross_entropy(o,t)	
