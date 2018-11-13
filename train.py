from vae import VAE
from utils import TrainSet
import torch
import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from os.path import join as opj
from os.path import expanduser as ope

import argparse

parser = argparse.ArgumentParser(description='arguments for VAE trainer')
parser.add_argument('--root', type=str, default=opj(ope('~'), 'Documents', 'MSRA-TD500', 'trainim'),
                    help='datasets root for training VAE (default MSRA-TD500)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size for training')
parser.add_argument('--loader', type=str, default='PIL.color',
                    help='datasets loader for training')
parser.add_argument('--resize', type=int, default=28,
                    help='resize for training images')
parser.add_argument('--interval', type=int, default=1000,
                    help='save per interval iterations')
opts = parser.parse_args()

def train(loader, model):
	
	for cnt, (data, _) in enumerate(loader):
		# import ipdb
		# ipdb.set_trace()
		decoded, mu, logvar = model(data)

if __name__ == '__main__':
	
	MODEL = VAE(nz=20, nc=3, ndf=64, ngf=64, nlatent=2048)
	
	TRAINSET = TrainSet(root=opts.root, loader=opts.loader,
	                    transform=transforms.Compose([
		                          transforms.Resize((opts.resize, opts.resize)),
		                          transforms.ToTensor()]))
	
	LOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=opts.batch_size)
	
	train(loader=LOADER, model=MODEL)