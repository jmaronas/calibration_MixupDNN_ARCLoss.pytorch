import torch
if torch.__version__!='1.0.0':
        raise Exception('Pytorch version must be 1.0.0')
import torch.nn as nn

from utils_cost import accuracy_per_bin,average_confidence_per_bin,confidence_per_bin

#the different calibration wrappers to the different newtorks
class KUMAR_net(nn.Module):
	def __init__(self, model_wrapped,lamda):
		super(KUMAR_net, self).__init__()
		self.model=model_wrapped
		self.lamda=lamda
	def forward(self,x):
		return self.model.forward(x)

	def forward_test(self,x):
		return self.model.forward_test(x)


	def classification_error(self,out,t):
		return (out.argmax(dim=1)!=t).sum()


	def classification_acc(self,out,t):
		return (out.argmax(dim=1)==t).sum()

	def cost_LLH(self,o,t):
		return nn.functional.cross_entropy(o,t)


	def kernel(self,ri,rj):
		return torch.exp(-1.0*torch.abs(ri-rj)/0.4) #same kernel as Kumar et al


	def cost_KUMAR(self,logit,t):
		probs=nn.functional.softmax(logit,dim=1)
		confidence,target=probs.max(dim=1)

		index_correct = target==t
		index_incorrect = target!=t

		n,m=float(t.size(0)),float(index_correct.sum())

		correct_conf = confidence[index_correct] #they are vectors
		incorrect_conf = confidence[index_incorrect] 

		#reshape as matrix for efficience
		corr_conf=correct_conf.view(correct_conf.size(0),1)
		incorr_conf=incorrect_conf.view(incorrect_conf.size(0),1)

		cost_corr,cost_incorr,cost_corr_incorr=[0.0]*3
		#cost incorrect
		if len(incorr_conf)!=0:
			w_incorr = incorr_conf.mm(incorr_conf.transpose(0,-1))
			incorr_conf_rep=incorr_conf.repeat(1,incorr_conf.size(0))
			ker_incorr = self.kernel(incorr_conf_rep,incorr_conf_rep.transpose(0,-1))
			cost_incorr =((w_incorr*ker_incorr)/((n-m)**2+1e-11)).sum()

		#cost correct
		if len(corr_conf)!=0:
			w_corr =  torch.mm((1-corr_conf),(1-corr_conf).transpose(0,-1))
			corr_conf_rep=corr_conf.repeat(1,corr_conf.size(0))
			ker_corr = self.kernel(corr_conf_rep,corr_conf_rep.transpose(0,-1))
			cost_corr = ((w_corr*ker_corr)/(m**2+1e-11)).sum()

		#cost correct incorrect
		if len(corr_conf)!=0 and len(incorr_conf)!=0:
			w_corr_incorr =  torch.mm((1-corr_conf),incorr_conf.transpose(0,-1))
			corr_conf_rep=corr_conf.repeat(1,incorr_conf.size(0))
			incorr_conf_rep=incorr_conf.repeat(1,corr_conf.size(0))
			ker_corr_incorr = self.kernel(corr_conf_rep,incorr_conf_rep.transpose(0,-1))
			cost_corr_incorr = ((w_corr_incorr*ker_corr_incorr)/((n-m)*m+1e-11)).sum()

		COST=torch.sqrt(cost_corr+cost_incorr-2*cost_corr_incorr)

		return self.lamda*COST


