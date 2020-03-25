# -*- coding: utf-8 -*-
# Author: Juan Maroñas (jmaronasm@gmail.com) PRHLT Research Center

#python
import numpy
import math
import logging
import time
import sys
import os
sys.path.extend([os.path.join(os.path.expanduser('~'),'pytorch_library/')])

#pytorch
import torch
if torch.__version__!='1.0.0':
	raise Exception('Pytorch version must be 1.0.0')
torch.manual_seed(seed=1)
torch.backends.cudnn.benchmark=True

#mine
from networks import CalibIdea_NoBins,CalibIdea_Bins
from data import load_data,batch_test,batch_train
from network_config import load_network
from utils import parse_args_ARC, compute_calibration_measures
from SGD import load_SGD_params
from pytorch_library import  add_experiment_notfinished,add_nan_file,remove_experiment_notfinished,save_checkpoint

######MAIN########
args=parse_args_ARC()

#create dataloaders
train_loader,_,valid_loader,test_loader,data_stats=load_data(args,valid_set_is_replicated=False)
total_train_data,total_test_data,total_valid_data,n_classes = data_stats

#load the network
if args.cost_type=='square[avgconf_sub_acc]':                                            
        calib_cost_type='square_avgconf_sub_acc'                                         
        calib_cost_index=1
elif args.cost_type=='avg[square[conf_sub_acc]]':                                        
        calib_cost_type='avg_square_sub_acc'                                             
        calib_cost_index=2
        
pretrained = True if args.dataset =='birds' or args.dataset=='cars' else False #we use pretrained models on imagenet
net,params=load_network(args.model_net,pretrained,args.n_gpu,n_classes=n_classes,dropout=args.dropout)

bins_for_train = [int(i) for i in args.bins_for_train]
if len(bins_for_train)==1 and bins_for_train[0] == 1:                           
	net=CalibIdea_NoBins(net,calib_cost_index,args.lamda)
else:
	net=CalibIdea_Bins(net,bins_for_train,calib_cost_index,args.lamda)          
net.cuda()

#usefull variables to monitor calibration error
predictions_train=torch.from_numpy(numpy.zeros((total_train_data,n_classes),dtype='float32'))
labels_train=torch.from_numpy(numpy.zeros((total_train_data,),dtype='int64'))
predictions_valid=torch.from_numpy(numpy.zeros((total_valid_data,n_classes),dtype='float32'))
labels_valid=torch.from_numpy(numpy.zeros((total_valid_data,),dtype='int64'))
predictions_test=torch.from_numpy(numpy.zeros((total_test_data,n_classes),dtype='float32'))
labels_test=torch.from_numpy(numpy.zeros((total_test_data,),dtype='int64'))
bins_for_eval=15


#to save the model and to perform logging
best_test=1e+14
bins_list_dir = "_".join(args.bins_for_train)                                            
valid_name = './checkpoint/validation/ARC/' if args.use_valid_set else './checkpoint/test/ARC/'
model_log_dir = os.path.join(valid_name,calib_cost_type+"_"+str(calib_cost_index)+"_lamda_"+str(args.lamda)+"_bins_"+bins_list_dir,args.dataset,args.model_net+"_drop"+str(args.dropout))

model_log_dir+="/" # need this to solve small bug in my library

try:
	os.makedirs(model_log_dir)
except:
	if not args.DEBUG:
		raise Exception("This directory already exists")


add_experiment_notfinished(model_log_dir)
logging.basicConfig(filename=model_log_dir+'train.log',level=logging.INFO)
logging.info("ARC Loss")
logging.info("Model: {} calibration error measured with {} bins".format(args.model_net+"_drop"+str(args.dropout),bins_for_eval))
logging.info("Calib cost type {} with cost index {} bins for training {} lamda {}".format(calib_cost_type,calib_cost_index,args.bins_for_train,str(args.lamda)))
logging.info("Batch size train {} total train {} total valid {} total test {} ".format(batch_train,total_train_data,total_valid_data,total_test_data))

#Stochastic Gradient Descent parameters and stuff
num_epochs,lr_init,wd,lr_scheduler = load_SGD_params(args.model_net,args.dataset)
parameters_fc,parameters_conv=params

for ep in range(num_epochs):
	random_lr=lr_scheduler(lr_init,ep+1,num_epochs)
	SGD_fc=torch.optim.SGD(parameters_fc,lr=random_lr,momentum=0.9,weight_decay=wd)
	conv_lr=random_lr/10. if ep > -1 else 0.0
	if args.dataset=='birds' or args.dataset=='cars':
		SGD_conv=torch.optim.SGD(parameters_conv,lr=conv_lr,momentum=0.9,weight_decay=wd)	
		SGD=[SGD_conv,SGD_fc]
	else:
		SGD=[SGD_fc]
	SSE_loss,CE_loss,MC_train,MC_test,MC_valid,total_test,total_valid,total_train,total_batch=[0.0]*9
	current_t=time.time()
	for idx,(x,t) in enumerate(train_loader):
		x,t=x.cuda(),t.cuda()

		out=net.forward(x)

		NNL=net.cost_LLH(out,t)
		SSE=net.cost_CONF(out,t)

		CE_loss+=NNL.data
		SSE_loss+=SSE.data

		cost=NNL+SSE

		cost.backward()

		for op in SGD:
			op.step()
			op.zero_grad()

		MC_train+=net.classification_error(out,t).data
		total_train+=t.size(0)
		total_batch+=1

		predictions_train[idx*batch_train:idx*batch_train+batch_train,:]=out.data.cpu()
		labels_train[idx*batch_train:idx*batch_train+batch_train]=t.data.cpu()
		print('| Epoch [{}/{}] Iter[{:.0f}/{:.0f}]\t\t Loss: {:.3f}'.format(ep+1,num_epochs,total_batch,len(train_loader),cost.data),end='\r')

	print("\n")
	with torch.no_grad():
		for idx,(x,t) in enumerate(valid_loader):
			x,t=x.cuda(),t.cuda()
			out=net.forward_test(x)
			MC_valid+=net.classification_error(out,t)
			total_valid+=t.size(0)
			predictions_valid[idx*batch_test:idx*batch_test+batch_test,:]=out.data.cpu()
			labels_valid[idx*batch_test:idx*batch_test+batch_test]=t.data.cpu()
		for idx,(x,t) in enumerate(test_loader):
			x,t=x.cuda(),t.cuda()
			out=net.forward_test(x)
			MC_test+=net.classification_error(out,t)
			total_test+=t.size(0)
			predictions_test[idx*batch_test:idx*batch_test+batch_test,:]=out.data.cpu()
			labels_test[idx*batch_test:idx*batch_test+batch_test]=t.data.cpu()
	
		'''
		Monitoring Calibration Error
		'''
		ECEtrain,MCEtrain,BRIERtrain,NNLtrain=compute_calibration_measures(predictions_train,labels_train,apply_softmax=True,bins=bins_for_eval)
		ECEtest,MCEtest,BRIERtest,NNLtest=compute_calibration_measures(predictions_test,labels_test,apply_softmax=True,bins=bins_for_eval)
		ECEvalid,MCEvalid,BRIERvalid,NNLvalid=[0.0]*4
		if args.use_valid_set:
			ECEvalid,MCEvalid,BRIERvalid,NNLvalid=compute_calibration_measures(predictions_valid,labels_valid,apply_softmax=True,bins=bins_for_eval)

		'''variables to display'''
		train_error=float(MC_train)/total_train*100
		valid_error=float(MC_valid)/total_valid*100 if args.use_valid_set else 0
		test_error=float(MC_test)/total_test*100
		CE_loss*=1/total_batch
		SSE_loss*=1/total_batch

	
	print("|| Epoch {} took {:.1f} minutes \tLossCE {:.5f}   LossSSE {:.5f}\n"
               "| Accuracy statistics:  Err train:{:.3f}  Err valid:{:.3f}  Err test:{:.3f} \n"
               "| Calibration Train:  ECE:{:.5f} MCE:{:.5f} BRIER:{:.5f} NNL:{:.5f} \n"
               "| Calibration valid: ECE:{:.5f} MCE:{:.3f} BRIER:{:.3f} NNL:{:.5f} \n"
               "| Calibration test: ECE:{:.5f} MCE:{:.5f} BRIER:{:.5f}  NNL:{:.5f}\n"
	       .format(ep, (time.time()-current_t)/60.,CE_loss,SSE_loss, train_error,valid_error,test_error,
                                                ECEtrain*100,MCEtrain,BRIERtrain,NNLtrain,
                                                ECEvalid*100,MCEvalid,BRIERvalid,NNLvalid,
                                                ECEtest*100,MCEtest,BRIERtest,NNLtest,
                                                ))

	logging.info("|| Epoch {} took {:.1f} minutes \tLossCE {:.5f}   LossSSE {:.5f}\n"
               "| Accuracy statistics:  Err train:{:.3f}  Err valid:{:.3f}  Err test:{:.3f} \n"
               "| Calibration Train:  ECE:{:.5f} MCE:{:.5f} BRIER:{:.5f} NNL:{:.5f} \n"
               "| Calibration valid: ECE:{:.5f} MCE:{:.3f} BRIER:{:.3f} NNL:{:.5f} \n"
               "| Calibration test: ECE:{:.5f} MCE:{:.5f} BRIER:{:.5f}  NNL:{:.5f}\n"
	       .format(ep, (time.time()-current_t)/60.,CE_loss,SSE_loss, train_error,valid_error,test_error,
                                                ECEtrain*100,MCEtrain,BRIERtrain,NNLtrain,
                                                ECEvalid*100,MCEvalid,BRIERvalid,NNLvalid,
                                                ECEtest*100,MCEtest,BRIERtest,NNLtest,
                                                ))

	if torch.isnan(cost):
		add_nan_file(model_log_dir)
		exit(-1)

	if MC_test <= best_test:
		save_checkpoint({
		'state_dict' : net.state_dict()},
		directory=model_log_dir,
		is_best=False,
		filename='model_best.pth'
		)
		best_test=MC_test


save_checkpoint({
		'state_dict' : net.state_dict()},
		directory=model_log_dir,
		is_best=False,
		filename='model.pth'
		)
remove_experiment_notfinished(model_log_dir)
exit(0)








