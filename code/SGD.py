# -*- coding: utf-8 -*-
# Author: Juan Maroñas (jmaronasm@gmail.com) PRHLT Research Center

import sys
import os
import math
sys.path.extend([os.path.expanduser('~/pytorch_library/')])
from pytorch_library import anneal_lr

def learning_rate_cifar10_cifar100_wideresnet(init, epoch,total_epoch=0):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def learning_rate_cifar10_cifar100_densenet(init, epoch,total_epoch=0):
    if(epoch > 250):
        lr=0.001
    elif(epoch > 150):
        lr=0.01
    else:
        lr=0.1

    return lr

def learning_rate_cifar10_cifar100_resnet(init, epoch,total_epoch=0):
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    return lr

def learning_rate_cifar10_cifar100_mobilenetv2(init, epoch,total_epoch):
    if(epoch < 80):
        lr=0.1
    elif(epoch < 150):
        lr=0.01
    else:
        lr = anneal_lr(0.01,total_epoch,epoch-1)

    return lr


def learning_rate_birds_cars_densenet(init, epoch,total_epoch=0):
    if(epoch > 250):
        lr=0.0001
    elif(epoch > 150):
        lr=0.001
    else:
        lr=0.01

    return lr

def learning_rate_birds_cars_resnet(init, epoch,total_epoch=0):
    if epoch < 80:
        lr = 0.01
    elif epoch < 120:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


def learning_rate_mnist_LeNet(init, epoch):

    if(epoch > 100):
        lr=init*0.1	
    else:
        lr=init
    return lr

def load_SGD_params(network_name,dataset):

	if 'wideresnet'==network_name.split("-")[0]:
		if dataset=='cifar10' or dataset=='cifar100' or dataset=='svhn':
			num_epochs = 200
			lr_init=0.1
			wd=5e-4
			lr_scheduler = learning_rate_cifar10_cifar100_wideresnet

	if 'densenet'==network_name.split("-")[0]:
		if dataset=='cifar10' or dataset=='cifar100' or dataset=='svhn':
			num_epochs = 350
			lr_init=0.1
			wd=5e-4
			lr_scheduler = learning_rate_cifar10_cifar100_densenet

		if dataset=='birds' or dataset=='cars':
			num_epochs = 350
			lr_init=0.01
			wd=5e-4
			lr_scheduler = learning_rate_birds_cars_densenet

	if 'resnet'==network_name.split("-")[0]:
		if dataset=='cifar10' or dataset=='cifar100' or dataset=='svhn':
			num_epochs = 200
			lr_init=0.1
			wd=1e-4
			lr_scheduler = learning_rate_cifar10_cifar100_resnet

		if dataset=='birds' or dataset=='cars':
			num_epochs = 200
			lr_init=0.01
			wd=1e-4
			lr_scheduler = learning_rate_birds_cars_resnet

	if 'mobilenetv2'==network_name:
		if dataset=='cifar10' or dataset=='cifar100' or dataset=='svhn' or dataset=='birds' or dataset=='cars':
			num_epochs = 350
			lr_init=0.1
			wd=5e-4
			lr_scheduler = learning_rate_cifar10_cifar100_mobilenetv2

	if 'squeezenet1_1'==network_name:
		if dataset=='birds' or dataset=='cars':
			num_epochs = 350
			lr_init=0.01
			wd=5e-4
			lr_scheduler = learning_rate_birds_cars_densenet

	return num_epochs,lr_init,wd,lr_scheduler
