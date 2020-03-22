import torch
import torch.nn as nn
from torch.nn.functional import softmax

import numpy

def accuracy_per_bin(predicted,real_tag,n_bins=15,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	

	accuracy,index = torch.max(predicted_prob,1)
	selected_label=index.long()==real_tag

	prob=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	acc=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	total_data = len(accuracy)
	samples_per_bin=[]
	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down=accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down=accuracy>min_

		index_range=boolean_down & boolean_upper
		label_sel=selected_label[index_range]
		
		if len(label_sel)==0:
			acc[p]=0.0
		else:
			acc[p]=label_sel.sum().float()/float(len(label_sel))

		samples_per_bin.append(len(label_sel))

	samples_per_bin=torch.from_numpy(numpy.array(samples_per_bin)).cuda()
	acc=acc[0:-1]
	prob=prob[0:-1]
	return acc,prob,samples_per_bin


def average_confidence_per_bin(predicted,n_bins=15,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	
	prob=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	conf=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	accuracy,index = torch.max(predicted_prob,1)

	samples_per_bin=[]
	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		
		boolean_upper = accuracy<=max_

		if p==0:#we include the first element in bin
			boolean_down =accuracy>=min_
		else:#after that we included in the previous bin
			boolean_down =accuracy>min_

		index_range=boolean_down & boolean_upper
		prob_sel=accuracy[index_range]
		
		if len(prob_sel)==0:
			conf[p]=0.0
		else:
                        conf[p]=prob_sel.sum().float()/float(len(prob_sel))

		samples_per_bin.append(len(prob_sel))

	samples_per_bin=torch.from_numpy(numpy.array(samples_per_bin)).cuda()
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf,prob,samples_per_bin

def confidence_per_bin(predicted,n_bins=10,apply_softmax=True):

	if apply_softmax:
		predicted_prob=softmax(predicted,dim=1)
	else:
		predicted_prob=predicted
	
	prob=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	conf=torch.from_numpy(numpy.linspace(0,1,n_bins+1)).float().cuda()
	max_confidence,index = torch.max(predicted_prob,1)

	samples_per_bin=[]
	conf_values_per_bin=[]

	for p in range(len(prob)-1):
		#find elements with probability in between p and p+1
		min_=prob[p]
		max_=prob[p+1]
		
		boolean_upper = max_confidence<=max_

		if p==0:#we include the first element in bin
			boolean_down =max_confidence>=min_
		else:#after that we included in the previous bin
			boolean_down =max_confidence>min_

		index_range=boolean_down & boolean_upper
		prob_sel=max_confidence[index_range]
		
		if len(prob_sel)==0:
			conf_values_per_bin.append([0.0])
		else:
			conf_values_per_bin.append(prob_sel)

		samples_per_bin.append(len(prob_sel))

	samples_per_bin=torch.from_numpy(numpy.array(samples_per_bin)).cuda()
	conf=conf[0:-1]
	prob=prob[0:-1]

	return conf_values_per_bin,prob,samples_per_bin

