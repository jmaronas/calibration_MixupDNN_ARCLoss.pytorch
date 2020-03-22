
import torch
if torch.__version__!='1.0.0':
        raise Exception('Pytorch version must be 1.0.0')

import torch.nn as nn
import torch.nn.functional as F


class distributed_Net(nn.Module):
	def __init__(self, wrapped_net,devices):
		super(distributed_Net, self).__init__()
		self.wrapped_net = nn.DataParallel(wrapped_net,devices)

	def forward(self, x):
		self.train()
		return self.wrapped_net(x)

	def forward_test(self, x):
		self.eval()
		return self.wrapped_net(x)

	
