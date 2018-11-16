from tqdm import tqdm
import os
import shutil
import json

from torch.utils.data.dataset import Dataset

from PIL import Image
import cv2

def clustering(NUM_OF_CLUSTERING, IMG_ROOT, ROOT, ID, MODEL=None, DATASET=None):
	
	pbar = tqdm(total=len(os.listdir(IMG_ROOT)))
	
	
	'''
	if not ((MODEL == None) and (DATASET == None)):
		cluster_path = os.path.join(ROOT, MODEL + '_' + DATASET + '_clustering')
		index_path = os.path.join(ROOT, MODEL + '_' + DATASET + '.index')
	elif not NONDEEP == None:
		cluster_path = os.path.join(ROOT, 'NON_DEEP_' + NONDEEP + '_clustering')
		index_path = os.path.join(ROOT, 'NON_DEEP_' + NONDEEP + '.index')
	else:
		raise IOError
	'''

	if (MODEL == None) or (DATASET == None):
		raise IOError
	else:
		cluster_path = os.path.join(ROOT, MODEL + '_' + DATASET + '_clustering')
		index_path = os.path.join(ROOT, MODEL + '_' + DATASET + '.index')
	
	
	with open(index_path, 'r') as f:
		index = json.load(f)
	
	if os.path.exists(cluster_path):
		shutil.rmtree(cluster_path)
		os.makedirs(cluster_path)
	else:
		os.makedirs(cluster_path)
	
	for k in range(NUM_OF_CLUSTERING):
		os.mkdir(os.path.join(cluster_path, str(k)))
	
	for id, idx in enumerate(ID):
		cpath = os.path.join(cluster_path, str(idx))
		
		src = os.path.join(IMG_ROOT, index[id])
		shutil.copy(src, cpath)
		pbar.update(1)
	
	pbar.close()


class TrainSet(Dataset):
	def __init__(self, root, loader='PIL', transform=None):
		self.root = root
		self.ims = self.__filter(root)
		# transform parameter only use for PIL loader,
		# for opencv, resize to (224, 224) by default
		self.transform = transform

		self.loader = loader.split('.')[0]
		
		self.is_color = 0
		if len(loader.split('.')) == 2:
			self.is_color = 1

		if self.loader == 'PIL':
			im_loader = self.__default_loader
		elif self.loader == 'opencv':
			im_loader = self.__cv_loader
		else:
			raise NotImplementedError
	
	def __getitem__(self, index, RESIZE=(224, 224)):
		## FIXME modified for one-file features
		image_path = os.path.join(self.root, self.ims[index])
		
		if self.loader == 'PIL':
			img = self.__default_loader(image_path)
			if self.transform is not None:
				return self.transform(img), self.ims[index]
			else:
				return img, self.ims[index]
		elif self.loader == 'opencv':
			# resize for resnet or vgg
			img = self.__cv_loader(image_path, resize=RESIZE, color=self.is_color)
			if self.is_color:
				img = img.transpose(2, 1, 0)    # reshape into (n_channel, w, h)
			return img, self.ims[index]
		else:
			raise NotImplementedError
	
	def __len__(self):
		return len(self.ims)
	
	@classmethod
	def __filter(cls, root):
		img_files = []
		files = os.listdir(root)
		for file in files:
			if os.path.basename(file).split('.')[1].lower() == 'jpg':
				img_files.append(file)
		
		return img_files
	
	@classmethod
	def __default_loader(cls, x):
		return Image.open(x).convert("RGB")
	
	@classmethod
	def __cv_loader(cls, x, resize, color):
		if not isinstance(resize, tuple):
			resize = (resize, resize)
		im = cv2.imread(x, color)
		return cv2.resize(im, resize)


