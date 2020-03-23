# -*- coding: utf-8 -*-

#torch
import torch
if torch.__version__!='1.0.0':
	raise Exception('Pytorch version must be 1.0.0')
from torch.nn.functional import softmax
import torchvision.models as models

import numpy
import argparse
import os
import sys
sys.path.extend([os.path.expanduser('~'),'pytorch_library'])

from pytorch_library import accuracy_per_bin,average_confidence_per_bin,compute_ECE,compute_MCE,compute_brier,categorical_to_one_hot,confidence_per_bin


def compute_calibration_measures(predictions,true_labels,apply_softmax=True,bins=15):

	predictions = softmax(predictions,1) if apply_softmax else predictions

	''' ECE and MCE'''
	acc_bin,prob,samples_per_bin=accuracy_per_bin(predictions,true_labels,n_bins=bins,apply_softmax=False)
	conf_bin,prob,samples_per_bin=average_confidence_per_bin(predictions,n_bins=bins,apply_softmax=False)
	ECE,_=compute_ECE(acc_bin,conf_bin,samples_per_bin)
	MCE,_=compute_MCE(acc_bin,conf_bin,samples_per_bin)
	
	'''Brier Score'''
	max_val=predictions.size(1)
	t_one_hot=categorical_to_one_hot(true_labels,max_val)
	BRIER=compute_brier(predictions,t_one_hot)

	''' NNL '''
	NNL=((t_one_hot*(-1*torch.log(predictions))).sum(1)).mean()

	return ECE,MCE,BRIER,NNL


def parse_args_baseline():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--DEBUG',default=False,action='store_true',help='Debug mode')

	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args

def parse_args_mixup():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--mixup_coeff',default=1,type=float,help='mixup coefficient')
	parser.add_argument('--DEBUG',default=False,action='store_true',help='Debug mode')

	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args


def parse_args_ARC():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--cost_type', type=str,default=None,choices=['square[avgconf_sub_acc]','avg[square[conf_sub_acc]]'],required=False,help='Type_of_cost_used')
	parser.add_argument('--lamda', type=float,required=True,help='hyperparameter controlling the importance of the Calibration cost')
	parser.add_argument('--bins_for_train',nargs='+',default=None,required=False,help='A list containing how many bins do you want to use to partition your confidence line')
	parser.add_argument('--DEBUG',default=False,action='store_true',help='Debug mode')

	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args


def parse_args_MMCE():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--lamda',default=1,type=float,help='lamda coefficient')
	parser.add_argument('--DEBUG',default=False,action='store_true',help='Debug mode')

	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args


def parse_args_ARC_mixup():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--mixup_coeff',default=1,type=float,help='mixup coefficient')
	parser.add_argument('--cost_type', type=str,default=None,choices=['square[avgconf_sub_acc]','avg[square[conf_sub_acc]]'],required=False,help='Type_of_cost_used')
	parser.add_argument('--bins_for_train',nargs='+',default=None,required=False,help='A list containing how many bins do you want to use to partition your confidence line')
	parser.add_argument('--lamda', type=float,required=True,help='hyperparameter controlling the importance of the Calibration cost')
	parser.add_argument('--cost_over_mix_image', type=int,required=True,choices=[0,1],help='hyperparameter controlling wether the proposed cost applies to the mixup image or not')
	parser.add_argument('--DEBUG',default=False,action='store_true',help='Debug mode')

	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args


def parse_args_MMCE_mixup():
	parser = argparse.ArgumentParser(description='Confidence calibration for Deep Neural Networks in Pytorch. Enjoy!')
	'''String Variables'''
	parser.add_argument('--model_net', type=str,choices=['wideresnet-16x8','wideresnet-28x10','wideresnet-40x10','densenet-121','resnet-101','mobilenetv2','squeezenet1_1','resnet-18','resnet-50'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['cifar10','cifar100','svhn','cars','birds'],required=True,help='dataset to use')
	parser.add_argument('--n_gpu', type=int,nargs='+',default=0,help='which gpu to use')
	parser.add_argument('--dropout', type=float,required=True, choices=[0.0,0.3,0.2,0.5],help='dropout value')
	parser.add_argument('--use_valid_set',default=1,choices=[0,1],type=int,help='Use a validation set')
	parser.add_argument('--lamda',default=1,type=float,help='lamda coefficient')
	parser.add_argument('--mixup_coeff',default=1,type=float,help='mixup coefficient')
	parser.add_argument('--cost_over_mix_image', type=int,required=True,choices=[0,1],help='hyperparameter controlling wether the proposed cost applies to the mixup image or not')
	args=parser.parse_args()
	torch.cuda.set_device(args.n_gpu[0])
	return args


