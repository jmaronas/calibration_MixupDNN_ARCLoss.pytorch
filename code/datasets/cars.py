# -*- coding: utf-8 -*-
# Author: Juan Maroñas (jmaronasm@gmail.com) PRHLT Research Center

'''
Standford cars dataset https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Requirements: scipy, tarfile, urrlib, pillow, progress,numpy,time,os,sys,torch,torchvision. I do not know what exactly the conda enviroments download.

Dataset Class. Main features (check class istance for wider explanation):
	-works as a standard torchvision dataset
	-you can either store the downloaded data in folders or in numpy arrays (depends on how much ram do you have)
	-possibility of returning bounding boxes
	-make some preprocess to the images, as they have different shapes
'''
# TODO: checksum

import torch
import torch.utils.data as data
from torchvision.datasets.folder import has_file_allowed_extension,IMG_EXTENSIONS,pil_loader,make_dataset

import os
import numpy
import time
import sys
import urllib.request
import tarfile
from PIL import Image
import scipy.io as sio
from scipy.misc import imresize,imsave
from progress.bar import Bar


def make_dataset_withbbox(dir, class_to_idx, extensions,bounding_box):#copied from github and modified to return boundingbox
	images = []
	dir = os.path.expanduser(dir)
	for target in sorted(class_to_idx.keys()):
		d = os.path.join(dir, target)
		if not os.path.isdir(d):
			continue


		for root, _, fnames in sorted(os.walk(d)):
			for fname in sorted(fnames):
				if has_file_allowed_extension(fname, extensions):
					idx=int(fname.split('.')[0])
					path = os.path.join(root, fname)
					item = (path, class_to_idx[target],bounding_box[idx,:])
					images.append(item)
	return images

class cars_standford(data.Dataset):

	def __init__(self,directory,asnumpy=False,download=True, transform=None, target_transform=None,isTrain=True,return_bounding_box=False,tam_image=300,interpolation='bilinear',padding='wrap'):
		'''A dataset loader for standford cars  https://ai.stanford.edu/~jkrause/cars/car_dataset.html

		Args:
			directory (string): Where the dataset is saved
			asnumpy (bool): either to store as numpy files or to store in folders. Files are stored following what torchvision.datasets.folder.DatasetFolder require. https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
			download (bool): download from internet or not
			transform (callable, optional): A function/transform that takes in
                            a sample and returns a transformed version.
		            E.g, ``transforms.RandomCrop`` for images.
		        target_transform (callable, optional): A function/transform that takes
			         in the target and transforms it.
			isTrain (bool): dataset for training or for test
			return_bounding_box (bool): prepare datase to return bounding box. If true asnumpy=False.
			padding (str): how to padd images to have same size. Please check numpy.pad for options. Default wrap. https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
			tam_image: the final image shape. Images are all padded to have size tam_imagextam_image
			interpolation: how to filter images to have the final desired tam_image. See scipy.misc imresize for options. Default bilinear. https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html
			
		'''

		self.directory=directory
		self.filename="car_ims"
		self.bbox_filename="cars_annos.mat"
		self.store_as_numpy=asnumpy
		self.padding=padding
		self.interpolation=interpolation
		self.tam_image=tam_image
		self.return_bounding_box=return_bounding_box
		self.istrain=isTrain

		aux=[str(i) for i in [self.tam_image,self.interpolation,self.padding]]
		extension= "/".join(aux)+"/"
		if self.return_bounding_box:
			self.directory_processed=self.directory+"processed/filefolder/boundingbox/"
		else:
			self.directory_processed=self.directory+"processed/filenumpy/"+extension if asnumpy else self.directory+"processed/filefolder/"+extension
		self.train_dir=self.directory_processed+"train/"
		self.test_dir=self.directory_processed+"test/"

		if download:
			self._download()
		else:
			if not self._file_exists(self.directory):
				raise RuntimeError('Dataset not found. You can use download=True to download it')

		self._check_if_process()
		self._prepare_for_loading()
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self,index):

		if self.return_bounding_box:
	
			path, target,bbox = self.samples[index]
			sample = pil_loader(path)
			if self.transform is not None:
				sample = self.transform(sample)
			if self.target_transform is not None:
				target = self.target_transform(target)

			return sample, target,bbox

		else:

			if self.store_as_numpy:
				sample = Image.fromarray(self.samples[index])
				target = self.targets[index]
			else:
				path, target = self.samples[index]
				sample = pil_loader(path)

			if self.transform is not None:
				sample = self.transform(sample)
			if self.target_transform is not None:
				target = self.target_transform(target)

			return sample, target

	def __len__(self):
		return len(self.samples)

	def default_collate(self,batch):
		'''Provide a collate option as images have different shapes cannot be returned in a torch.tensor, just pass instance.default_collate to the dataloader'''
		data=[item[0] for item in batch]
		target=[item[1] for item in batch]
		bbox=[item[2] for item in batch]
		return data,target,bbox

	def _prepare_for_loading(self):
		'''Prepare dataset to download.Depending on how the dataset is preprocessed (asnumpy, in folders, returnBoundingBoxes). It follows what torchvision.datasets.folder.DatasetFolder  does'''
		if self.return_bounding_box:
			name='cars_train_bbox.npz' if self.istrain else 'cars_test_bbox.npz'
			root = self.train_dir if self.istrain else self.test_dir
			bounding_box = numpy.load(self.directory_processed+name)['bbox']
			classes, class_to_idx = self._find_classes(root)
			samples = make_dataset_withbbox(root, class_to_idx, IMG_EXTENSIONS,bounding_box)
			self.samples = samples
			self.targets = [s[1] for s in samples]				

		elif not self.store_as_numpy:
			root = self.train_dir if self.istrain else self.test_dir
			classes, class_to_idx = self._find_classes(root)
			samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
			self.samples = samples
			self.targets = [s[1] for s in samples]				

		else:
			name='cars_train.npz' if self.istrain else 'cars_test.npz'			
			root = self.train_dir if self.istrain else self.test_dir
			data=numpy.load(root+name)
			self.samples=data['feat']
			self.targets=data['lab']

	def _uncompress(self):
		'''Uncompress downloaded file'''
		print("Uncompressing .tgz file")
		tar = tarfile.open(self.directory+self.filename+".tgz", "r:gz")
		tar.extractall(self.directory)
		tar.close()
		print("Finish")

	def _file_exists(self,directory):
		'''return true if file already exists'''
		return os.path.exists(directory)


	def _download(self):
		'''Download compressed file if it is not previously downloaded, and finally uncompresse it'''
		if self._file_exists(self.directory+self.filename):
			return

		print("Downloading standford cars. May take a while. 1.9 GB files to download...")
				
		url1='http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
		url2='http://imagenet.stanford.edu/internal/car196/car_ims.tgz'	
		try:
			os.makedirs(self.directory)
		except:
			pass
		urllib.request.urlretrieve(url2,self.directory+self.filename+".tgz")  		
		urllib.request.urlretrieve(url1,self.directory+self.bbox_filename)  		
		print("Finish.")

		self._uncompress()
		


	def _check_if_process(self):
		'''Check if file is already processed in the way specified by arguments isTrain=True,return_bounding_box=False,tam_image=300,interpolation='bilinear',padding='wrap'.'''
		if self._file_exists(self.directory_processed):
			print("File already download and processed")
			return
		else:
			self._processing()

	def _processing(self):
		'''Process files as specified by isTrain=True,return_bounding_box=False,tam_image=300,interpolation='bilinear',padding='wrap'''
		print("Processing data. May take long the first time...")
		bar = Bar('Processing', max=8144+8041, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

		file_bbox=self.directory+self.bbox_filename

		os.makedirs(self.train_dir)
		os.makedirs(self.test_dir)

		if not self.return_bounding_box:
			if self.store_as_numpy:
				train_database_feat = numpy.ones((8144,self.tam_image,self.tam_image,3),dtype=numpy.uint8)
				train_database_lab = -1*numpy.ones((8144,),dtype=numpy.int64)
				test_database_feat = numpy.ones((8041,self.tam_image,self.tam_image,3),dtype=numpy.uint8)
				test_database_lab = -1*numpy.ones((8041,),dtype=numpy.int64)

			else: 
				for i in range(196):		
					os.makedirs(self.train_dir+str(i))
					os.makedirs(self.test_dir+str(i))
		else:
			for i in range(196):		
				os.makedirs(self.train_dir+str(i))
				os.makedirs(self.test_dir+str(i))

			train_bbox = -1*numpy.ones((8144,4),dtype=numpy.float32)
			test_bbox = -1*numpy.ones((8041,4),dtype=numpy.float32)
	

		counter_train,counter_test=0,0
		for line in sio.loadmat(file_bbox)['annotations'][0]:
	
			image_name,a,b,c,d,label,is_test=line
			label=label[0,0]
			label=numpy.int64(label)-1
			
			x,y,width,height,is_test=int(a),int(b),int(c),int(d),int(is_test)

			image = numpy.array(Image.open(self.directory+image_name[0]))

			if not self.return_bounding_box:

				image = image[y:height,x:width]
				row=image.shape[0]
				col=image.shape[1]
				#padding
				if row>col:
					max_pad_col=row
					pad_size_col = (max_pad_col-col)/2.
					pad_size_col_l=numpy.ceil(pad_size_col).astype(numpy.int32)
					pad_size_col_r=numpy.floor(pad_size_col).astype(numpy.int32)
					pad_size_row_l,pad_size_row_r=0,0
				elif row<col:
					max_pad_row=col
					pad_size_row = (max_pad_row-row)/2.
					pad_size_row_l=numpy.ceil(pad_size_row).astype(numpy.int32)
					pad_size_row_r=numpy.floor(pad_size_row).astype(numpy.int32)
					pad_size_col_l,pad_size_col_r=0,0
				else:
					pad_size_row_l,pad_size_row_r,pad_size_col_l,pad_size_col_r=0,0,0,0

	
				if len(image.shape)==2:
					image = numpy.stack((image,)*3, -1)

				image=numpy.pad(image,((pad_size_row_l,pad_size_row_r),(pad_size_col_l,pad_size_col_r),(0,0)),self.padding)
				im_crop=imresize(image,(self.tam_image,self.tam_image,3),self.interpolation)


				if self.store_as_numpy:
					if not is_test:
						train_database_feat[counter_train]=im_crop
						train_database_lab[counter_train]=label
						counter_train+=1
					else:
						test_database_feat[counter_test]=im_crop
						test_database_lab[counter_test]=label
						counter_test+=1

				else:
					if not is_test:
						imsave(self.train_dir+str(label)+"/"+str(counter_train)+'.png', im_crop)
						counter_train+=1
					else:
						imsave(self.test_dir+str(label)+"/"+str(counter_test)+'.png', im_crop)	
						counter_test+=1

			else:
				if len(image.shape)==2: #some images are grayscale
					image = numpy.stack((image,)*3, -1)
				if not is_test:
					train_bbox[counter_train,:]=[x,y,width,height]
					imsave(self.train_dir+str(label)+"/"+str(counter_train)+'.png', image)
					counter_train+=1
				else:
					test_bbox[counter_test,:]=[x,y,width,height]
					imsave(self.test_dir+str(label)+"/"+str(counter_test)+'.png', image)	
					counter_test+=1
		
			bar.next()

		if self.store_as_numpy:
			numpy.savez(self.train_dir+'cars_train',feat=train_database_feat,lab=train_database_lab)
			numpy.savez(self.test_dir+'cars_test',feat=test_database_feat,lab=test_database_lab)

		if self.return_bounding_box:
			numpy.savez(self.directory_processed+'cars_train_bbox',bbox=train_bbox)
			numpy.savez(self.directory_processed+'cars_test_bbox',bbox=test_bbox)
		bar.finish()
		print("Finish processing data")	


	def _find_classes(self, dir):#copied from github torchvision.datasets.folder.DatasetFolder
		"""
		Finds the class folders in a dataset.
		Args:
		    dir (string): Root directory path.
		Returns:
		    tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
		Ensures:
		    No class is a subdirectory of another.
		"""
		if sys.version_info >= (3, 5):
		    # Faster and available in Python 3.5 and above
		    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
		else:
		    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		classes.sort()
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		return classes, class_to_idx

		
	


