from vae import VAE
from utils import TrainSet
import torch
from torch.nn import functional as F
import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from os.path import join as opj
from os.path import expanduser as ope

import argparse

parser = argparse.ArgumentParser(description='arguments for VAE trainer')
parser.add_argument('--root', type=str, default=opj(ope('~'), 'Documents', 'MSRA-TD500', 'trainim'),
                    help='datasets root for training VAE (default MSRA-TD500)')
parser.add_argument('--batch_size', type=int, default=5,
                    help='batch size for training')
parser.add_argument('--loader', type=str, default='PIL.color',
                    help='datasets loader for training')
parser.add_argument('--resize', type=int, default=512,
                    help='resize for training images')
parser.add_argument('--interval', type=int, default=1000,
                    help='save per interval iterations')
opts = parser.parse_args()


def loss_function(recon_x, x, bs, mu=None, logvar=None):
	# print(recon_x.size(), x.size())
	# BCE = F.binary_cross_entropy(recon_x.view(bs, -1), x.view(bs, -1))
	# import ipdb
	# ipdb.set_trace()
	BCE = F.binary_cross_entropy(recon_x, x)
	# import ipdb
	# ipdb.set_trace()
	# MSE = F.mse_loss(recon_x, x, size_average=False)
	# pixel_avg_bce = BCE/(x.shape[-1]*x.shape[-2])
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	# KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	# return BCE + KLD
	# return BCE + 3 * KLD
	# import ipdb
	# ipdb.set_trace()
	# return pixel_avg_bce + 5*KLD, pixel_avg_bce, KLD
	return BCE

def train(loader, model, optim):
	while(True):
		for cnt, (data, _) in enumerate(loader):
			# import ipdb
			# ipdb.set_trace()
			# decoded, mu, logvar = model(data)
			decoded = model(data)
			# loss, bce, kld = loss_function(decoded, data, mu, logvar, opts.batch_size)
			bce = loss_function(decoded, data, opts.batch_size)
			
			optim.zero_grad()
			# loss.backward()
			bce.backward()
			optim.step()
			
			print("training iter {}, loss {:.4f}".format(cnt+1, bce.item()))
		


if __name__ == '__main__':
	
	MODEL = VAE(nz=20, nc=3, ndf=64, ngf=64, nlatent=1024)
	
	TRAINSET = TrainSet(root=opts.root, loader=opts.loader,
	                    transform=transforms.Compose([
		                          transforms.Resize((opts.resize, opts.resize)),
		                          transforms.ToTensor()]))
	
	LOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=opts.batch_size)
	
	OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=0.0001)
	
	train(loader=LOADER, model=MODEL, optim=OPTIMIZER)