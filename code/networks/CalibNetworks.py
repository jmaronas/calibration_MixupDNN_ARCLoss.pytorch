## Author: Juan Maroñas PRHLT Research Center (jmaronas@prhlt.upv.es)
## Calibration wrappers

import torch
if torch.__version__!='1.0.0':
        raise Exception('Pytorch version must be 1.0.0')
import torch.nn as nn

from utils_cost import accuracy_per_bin,average_confidence_per_bin,confidence_per_bin

# Only M=1
class CalibIdea_NoBins(nn.Module):
	def __init__(self, model_wrapped,calib_cost_index,lamda):
		super(CalibIdea1, self).__init__()
		self.model=model_wrapped
		self.calib_cost_index=calib_cost_index
		self.lamda=lamda
	def forward(self,x):
		return self.model.forward(x)

	def forward_test(self,x):
		return self.model.forward_test(x)

	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()

	def cost_CONF(self,o,t):# \loss is applied over a single Image.
		COST=None
		acc=(self.classification_acc(o,t).float()/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
		predicted_prob=nn.functional.softmax(o,dim=1)
		predicted_prob,index = torch.max(predicted_prob,1)

		if self.calib_cost_index==1:
			COST=((predicted_prob.mean()-acc)**2)	
		elif self.calib_cost_index==2:
			COST=((predicted_prob-acc)**2).mean()

		return self.lamda*COST

	def cost_CONF_mixup_interpolated(self,o,t1,t2,lam):# \loss is applied over the mixup Image. 
		COST=None
		acc1=self.classification_acc(o,t1).float()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
		acc2=self.classification_acc(o,t2).float()#/float(t.size(0))).detach()#compute the accuracy, detach from the computation graph as we do not require this cost to modify this
		acc=(lam*acc1+(1-lam)*acc2)/float(t1.size(0))

		predicted_prob=nn.functional.softmax(o,dim=1)
		predicted_prob,index = torch.max(predicted_prob,1)

		if self.calib_cost_index==1:
			COST=((predicted_prob.mean()-acc)**2)	
		elif self.calib_cost_index==2:
			COST=((predicted_prob-acc)**2).mean()

		return self.lamda*COST

	def classification_acc(self,out,t):
		return (out.argmax(dim=1)==t).sum()

	def cost_LLH(self,o,t):
		return nn.functional.cross_entropy(o,t)


# Allow to select M \neq 1
class CalibIdea_Bins(nn.Module):
	def __init__(self, model_wrapped,bins_for_train,calib_cost_index,lamda):
		super(CalibIdea2, self).__init__()
		self.model=model_wrapped
		self.bins_for_train=bins_for_train
		self.calib_cost_index=calib_cost_index
		self.lamda=lamda
	def forward(self,x):
		return self.model.forward(x)

	def forward_test(self,x):	
		return self.model.forward_test(x)

	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()


	def calib_cost(self,a,c,c_per_bin,s,index):
		if index==1:
			return ((c-a.detach())**2)*s
		elif index==2:
			return (((c_per_bin-a.detach())**2).mean())*s			

	def cost_CONF(self,o,t):
		COST=torch.tensor(0.0).cuda()
		for n_bins in self.bins_for_train:
			aux_cost,tot_samples=[torch.tensor(0.0).cuda()]*2
			acc,prob,samples_per_bin=accuracy_per_bin(o,t,n_bins)
			avg_conf,prob,samples_per_bin=average_confidence_per_bin(o,n_bins) 
			conf_per_bins,prob,samples_per_bin=confidence_per_bin(o,n_bins) 

			for a,c,c_per_bin,s in zip(acc,avg_conf,conf_per_bins,samples_per_bin):
				if s!=0.0:
					aux_cost+=self.calib_cost(a,c,c_per_bin,s,self.calib_cost_index)

			if tot_samples!=0:		
				aux_cost*=1/tot_samples

			COST+=aux_cost

		COST*=self.lamda/float(len(self.bins_for_train))
		return COST
                

	def cost_CONF_mixup_interpolated(self,o,t1,t2,lam):
		COST=torch.tensor(0.0).cuda()
		for n_bins in self.bins_for_train:
			aux_cost,tot_samples=[torch.tensor(0.0).cuda()]*2
			acc1,prob,samples_per_bin1=accuracy_per_bin(o,t1,n_bins)
			acc2,prob,samples_per_bin2=accuracy_per_bin(o,t2,n_bins)
			acc=acc1.clone().detach()

			for idx,(a1,a2,s1,s2) in enumerate(zip(acc1,acc2,samples_per_bin1,samples_per_bin2)):
				if s1==0 and s2==0:
					acc[idx]=0.0
				else:
					a1,a2,s1,s2=a1.float(),a2.float(),s1.float(),s2.float()
					acc[idx]=(lam*a1*s1+(1-lam)*a2*s2)/float(lam*s1+(1-lam)*s2)
				
			avg_conf,prob,samples_per_bin=average_confidence_per_bin(o,n_bins) 
			conf_per_bins,prob,samples_per_bin=confidence_per_bin(o,n_bins) 

			for a,c,c_per_bin,s in zip(acc,avg_conf,conf_per_bins,samples_per_bin):
				if s!=0.0:
					aux_cost+=self.calib_cost(a,c,c_per_bin,s,self.calib_cost_index)

			if tot_samples!=0:		
				aux_cost*=1/tot_samples

			COST+=aux_cost

		COST*=self.lamda/float(len(self.bins_for_train))
		return COST
		
	def classification_acc(self,out,t):
		return (out.argmax(dim=1)==t).sum()

	def cost_LLH(self,o,t):
		return nn.functional.cross_entropy(o,t)


                
