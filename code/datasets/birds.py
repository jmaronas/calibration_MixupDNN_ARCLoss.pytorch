import torch
import torch.utils.data as data
from torchvision.datasets.folder import has_file_allowed_extension,IMG_EXTENSIONS,pil_loader,make_dataset

import urllib.request
import os
import tarfile
from PIL import Image
from scipy.misc import imresize,imsave
import numpy
import time
import sys
from progress.bar import Bar



def make_dataset_withbbox(dir, class_to_idx, extensions,bounding_box):
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
#TODO add checkmdsum
class birds_caltech_2011(data.Dataset):

	def __init__(self,directory, transform=None, target_transform=None,asnumpy=False,download=True,isTrain=True,tam_image=300,interpolation='bilinear',padding='wrap',return_bounding_box=False):
		'''
		Arguments
			-directory: where to save images
			-asnumpy: to store as numpy or in separate folders
			-download: whether to download or not
			-padding: how to padd images to have same size. Please check numpy.pad for options. Default wrap
			-tam_image: the final image shape. scipy.misc imresize for options 
			-interpolation: how to filter images to have the final desired tam_image. scipy.misc imresize for options
			-return_bounding_box: wether to return bounding box. In this case no interpolation nor padding is done. Also images are saved in folder and not as numpy images.
		'''

		self.directory=directory
		self.filename="CUB_200_2011"
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

			return sample,target,torch.from_numpy(bbox)

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
		data=[item[0] for item in batch]
		target=[item[1] for item in batch]
		bbox=[item[2] for item in batch]
		return data,target,bbox

	def _prepare_for_loading(self):
		if self.return_bounding_box:
			name='birds_train_bbox.npz' if self.istrain else 'birds_test_bbox.npz'
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
			name='birds_train.npz' if self.istrain else 'birds_test.npz'			
			root = self.train_dir if self.istrain else self.test_dir
			data=numpy.load(root+name)
			self.samples=data['feat']
			self.targets=data['lab']

	def _uncompress(self):
		print("Uncompressing .tgz file")
		tar = tarfile.open(self.directory+self.filename+".tgz", "r:gz")
		tar.extractall(self.directory)
		tar.close()
		print("Finish")

	def _file_exists(self,directory):
		return os.path.exists(directory)


	def _download(self):
		if self._file_exists(self.directory+self.filename):
			return

		print("Downloading birds_caltech_2011...")
		url='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'	
		try:
			os.makedirs(self.directory)
		except:
			pass
		urllib.request.urlretrieve(url,self.directory+self.filename+".tgz")  		
		print("Finish.")

		self._uncompress()


	def _check_if_process(self):
		if self._file_exists(self.directory_processed):
			print("File already download and processed")
			return
		else:
			self._processing()

	def _processing(self):
		print("Processing data. May take long the first time...")
		bar = Bar('Processing', max=5994+5794, suffix='Images %(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

		file_images=self.directory+self.filename+'/images.txt'
		file_train_test=self.directory+self.filename+'/train_test_split.txt'
		file_classes=self.directory+self.filename+'/image_class_labels.txt'
		file_bbox=self.directory+self.filename+'/bounding_boxes.txt'
		os.makedirs(self.train_dir)
		os.makedirs(self.test_dir)

		if not self.return_bounding_box:
			if self.store_as_numpy:
				train_database_feat = numpy.ones((5994,self.tam_image,self.tam_image,3),dtype=numpy.uint8)
				train_database_lab = -1*numpy.ones((5994,),dtype=numpy.int64)
				test_database_feat = numpy.ones((5794,self.tam_image,self.tam_image,3),dtype=numpy.uint8)
				test_database_lab = -1*numpy.ones((5794,),dtype=numpy.int64)

			else: 
				for i in range(200):		
					os.makedirs(self.train_dir+str(i))
					os.makedirs(self.test_dir+str(i))
		else:
			for i in range(200):		
				os.makedirs(self.train_dir+str(i))
				os.makedirs(self.test_dir+str(i))

			train_bbox = -1*numpy.ones((5994,4),dtype=numpy.float32)
			test_bbox = -1*numpy.ones((5794,4),dtype=numpy.float32)
	

		counter_train,counter_test=0,0
		for l1,l2,l3,l4 in zip(open(file_images),open(file_train_test),open(file_classes),open(file_bbox)):
			image_name=l1.split("\n")[0].split(" ")[1]
			is_train = int(l2.split("\n")[0].split(" ")[1])
			label    = numpy.int64(l3.split("\n")[0].split(" ")[1])-1

			x,y,width,height= l4.split("\n")[0].split(" ")[1:]
			x,y,width,height=numpy.float32(x).astype('int32'),numpy.float32(y).astype('int32'),numpy.float32(width).astype('int32'),numpy.float32(height).astype('int32')


			image = numpy.array(Image.open(self.directory+self.filename+'/images/'+image_name))
			
			if not self.return_bounding_box:
				image=image[y:y+height,x:x+width]
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

				if len(image.shape)==2: #some images are grayscale
					image = numpy.stack((image,)*3, -1)

				image=numpy.pad(image,((pad_size_row_l,pad_size_row_r),(pad_size_col_l,pad_size_col_r),(0,0)),self.padding)
				im_crop=imresize(image,(self.tam_image,self.tam_image,3),self.interpolation)	
				
				if self.store_as_numpy:
					if is_train:
						train_database_feat[counter_train]=im_crop
						train_database_lab[counter_train]=label
						counter_train+=1
					else:
						test_database_feat[counter_test]=im_crop
						test_database_lab[counter_test]=label
						counter_test+=1

				else:
					if is_train:
						imsave(self.train_dir+str(label)+"/"+str(counter_train)+'.png', im_crop)
						counter_train+=1
					else:
						imsave(self.test_dir+str(label)+"/"+str(counter_test)+'.png', im_crop)	
						counter_test+=1
			else:
				if len(image.shape)==2: #some images are grayscale
					image = numpy.stack((image,)*3, -1)
				if is_train:
					train_bbox[counter_train,:]=[x,y,width,height]
					imsave(self.train_dir+str(label)+"/"+str(counter_train)+'.png', image)
					counter_train+=1
				else:
					test_bbox[counter_test,:]=[x,y,width,height]
					imsave(self.test_dir+str(label)+"/"+str(counter_test)+'.png', image)	
					counter_test+=1
			bar.next()

		if self.store_as_numpy:
			numpy.savez(self.train_dir+'birds_train',feat=train_database_feat,lab=train_database_lab)
			numpy.savez(self.test_dir+'birds_test',feat=test_database_feat,lab=test_database_lab)

		if self.return_bounding_box:
			numpy.savez(self.directory_processed+'birds_train_bbox',bbox=train_bbox)
			numpy.savez(self.directory_processed+'birds_test_bbox',bbox=test_bbox)
		bar.finish()
		print("Finish processing data")	


	def _find_classes(self, dir):
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

		
	


