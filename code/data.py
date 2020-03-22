import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision as tv
import torchvision.transforms as tvtr

import numpy
import sys
import os
sys.path.extend([os.path.expanduser('~/pytorch_library/')])


from datasets import birds_caltech_2011,cars_standford

batch_test=500
batch_train=128

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'birds' : (0.5,0.5,0.5),
    'cars' : (0.5,0.5,0.5)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'birds' : (0.5,0.5,0.5),
    'cars' : (0.5,0.5,0.5)
}


def transforms(dataset):

	if dataset=='cifar10':
		return [tvtr.Compose([	
                       tvtr.RandomCrop(32, padding=4),
                       tvtr.RandomHorizontalFlip(),
                       tvtr.ToTensor(),
                       tvtr.Normalize(mean['cifar10'], std['cifar10'])]),
		 tvtr.Compose([tvtr.ToTensor(),tvtr.Normalize(mean['cifar10'], std['cifar10'])])
                ]
	elif dataset=='cifar100':
		return [tvtr.Compose([
                       tvtr.RandomCrop(32, padding=4),
                       tvtr.RandomHorizontalFlip(),
                       tvtr.ToTensor(),
                       tvtr.Normalize(mean['cifar100'], std['cifar100'])]),
		 tvtr.Compose([tvtr.ToTensor(),tvtr.Normalize(mean['cifar100'], std['cifar100'])])
                ]
	elif dataset=='svhn':
		return [tvtr.Compose([                    
                       tvtr.ToTensor()
                       ]),
		 tvtr.Compose([tvtr.ToTensor()])
                ]
	elif dataset=='birds':
		return [tvtr.Compose([
                       tvtr.RandomHorizontalFlip(),
                       tvtr.RandomResizedCrop(224),
                       tvtr.ToTensor(),
                       tvtr.Normalize(mean['birds'], std['birds'])]),
		 tvtr.Compose([tvtr.CenterCrop(224),tvtr.ToTensor(),tvtr.Normalize(mean['birds'], std['birds'])])
                 ]
	elif dataset=='cars':
		return [tvtr.Compose([
                       tvtr.RandomHorizontalFlip(),
                       tvtr.RandomResizedCrop(224),
                       tvtr.ToTensor(),
                       tvtr.Normalize(mean['cars'], std['cars'])]),
		 tvtr.Compose([tvtr.CenterCrop(224),tvtr.ToTensor(),tvtr.Normalize(mean['cars'], std['cars'])])
                ]

def datasets(dataset):
	traintr,testtr=transforms(dataset)

	if dataset=='cifar10':
	 	return [tv.datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=traintr),
                 tv.datasets.CIFAR10('./data/cifar10/',train=True,download=True,transform=testtr),
                 tv.datasets.CIFAR10('./data/cifar10/',train=False,download=True,transform=testtr),
		 45000,10000,5000,10]

	elif dataset=='cifar100':
		return [
                 tv.datasets.CIFAR100('./data/cifar100/',train=True,download=True,transform=traintr),
                 tv.datasets.CIFAR100('./data/cifar100/',train=True,download=True,transform=testtr),
                 tv.datasets.CIFAR100('./data/cifar100/',train=False,download=True,transform=testtr),
		 45000,10000,5000,100
                ]
	elif dataset=='svhn':
		return [
                 tv.datasets.SVHN('./data/svhn/',split='train',download=True,transform=traintr),
                 tv.datasets.SVHN('./data/svhn/',split='train',download=True,transform=testtr),
                 tv.datasets.SVHN('./data/svhn/',split='test',download=True,transform=testtr),
		 68257,26032,5000,10
                ]
	elif dataset=='birds':
		return [
                 birds_caltech_2011('./data/birds/',asnumpy=False,isTrain=True,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=traintr),
                 birds_caltech_2011('./data/birds/',asnumpy=False,isTrain=True,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=testtr),
                 birds_caltech_2011('./data/birds/',asnumpy=False,isTrain=False,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=testtr),
		 4994,5794,1000,200
                ]
	elif dataset=='cars':
		return [
                 cars_standford('./data/cars/',asnumpy=False,isTrain=True,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=traintr),
                 cars_standford('./data/cars/',asnumpy=False,isTrain=True,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=testtr),
                 cars_standford('./data/cars/',asnumpy=False,isTrain=False,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False,download=True,transform=testtr),
		 6184,8041,1960,196
                ]


def load_data(args,valid_set_is_replicated):
	#valid_set_is_replicated: -> this stands for replicating the valid dataset as to have as many samples as the training one

	'''Dataset and Data Loader'''
	train,valid,test,total_train_data,total_test_data,total_valid_data,n_classes=datasets(args.dataset)
	generator=numpy.random.RandomState(seed=1)
	random_index=generator.permutation(total_train_data+total_valid_data)

	if valid_set_is_replicated:
		untiled_valid_idx,train_idx=random_index[0:total_valid_data],random_index[total_valid_data:]
		aux=int(numpy.ceil(float(len(train_idx))/float(len(untiled_valid_idx))))
		valid_idx = numpy.tile(untiled_valid_idx,aux)

	else:
		if args.use_valid_set:
			untiled_valid_idx,valid_idx,train_idx=random_index[0:total_valid_data],random_index[0:total_valid_data],random_index[total_valid_data:]
		else:
			untiled_valid_idx,valid_idx,train_idx=numpy.array([]),numpy.array([]),random_index

	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	untiled_valid_sampler = SubsetRandomSampler(untiled_valid_idx)

	workers = (int)(os.popen('nproc').read()) 

	data_train =  torch.utils.data.DataLoader(train,batch_size=batch_train,sampler=train_sampler,num_workers=workers)
	data_valid =  torch.utils.data.DataLoader(valid,batch_size=batch_train,sampler=valid_sampler,num_workers=int(workers))

	data_valid_untiled =  torch.utils.data.DataLoader(valid,batch_size=batch_test,sampler=untiled_valid_sampler,num_workers=int(workers))
	data_test =  torch.utils.data.DataLoader(test,batch_size=batch_test,shuffle=False,num_workers=int(workers))

	if not args.use_valid_set:
		total_train_data+=total_valid_data
		total_valid_data = 0

	return data_train,data_valid,data_valid_untiled,data_test,[total_train_data,total_test_data,total_valid_data,n_classes]



