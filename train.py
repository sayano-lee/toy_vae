from vae import VAE
from utils import TrainSet
import torch
from torch.nn import functional as F
import torchvision
from torchvision.utils import save_image
from torchvision import transforms


from tensorboardX import SummaryWriter


from os.path import join as opj
from os.path import expanduser as ope

import argparse

parser = argparse.ArgumentParser(description='arguments for VAE trainer')

parser.add_argument('--root', type=str, default=opj(ope('~'), 'Documents', 'datasets', 'MSRA-TD500', 'trainim'),
                    help='datasets root for training VAE (default MSRA-TD500)')
parser.add_argument('--batch_size', type=int, default=15,
                    help='batch size for training')
parser.add_argument('--loader', type=str, default='PIL.color',
                    help='datasets loader for training')
parser.add_argument('--resize', type=int, default=256,
                    help='resize for training images')
parser.add_argument('--interval', type=int, default=1000,
                    help='save model per interval iterations')
parser.add_argument('--log_interval', type=int, default=5, help='save log interval for tensorboard')
parser.add_argument('--log_path', type=str, default=opj(ope('~'), 'codes', 'toy_vae', 'tensorboard'))
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate for optimizer')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

opts = parser.parse_args()

opts.device = None
if not opts.disable_cuda and torch.cuda.is_available():
    opts.device = torch.device('cuda')
else:
    opts.device = torch.device('cpu')

def loss_function(recon_x, x, bs, mu=None, logvar=None):
	# print(recon_x.size(), x.size())
	# BCE = F.binary_cross_entropy(recon_x.view(bs, -1), x.view(bs, -1))
	# import ipdb
	# ipdb.set_trace()

	# BCE = F.binary_cross_entropy(recon_x, x)
	MSE = F.mse_loss(recon_x, x)


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

	return MSE

def train(loader, model, optim, writer):
	epoch = 0
	count = 0
	length = len(loader)
	# import ipdb
	# ipdb.set_trace()
	while(True):

		for cnt, (data, _) in enumerate(loader):
			count += 1
			data = data.to(opts.device)
			decoded = model(data)

			loss = loss_function(decoded, data, opts.batch_size)
			
			optim.zero_grad()
			loss.backward()
			optim.step()

			if (cnt+1) % opts.log_interval == 0:
				writer.add_scalar('loss', float('{:.4f}'.format(loss.item())), count)
				# writer.add_scalar('loss', loss.item(), cnt)

			print("training iter {} in epoch {}, loss {:.4f}".format(cnt, epoch, loss.item()))
			epoch += 1


if __name__ == '__main__':
	
	MODEL = VAE(nz=20, nc=3, ndf=64, ngf=64, nlatent=1024)

	MODEL.to(opts.device)
	# MODEL.cuda()
	# import ipdb
	# ipdb.set_trace()
	
	TRAINSET = TrainSet(root=opts.root, loader=opts.loader,
	                    transform=transforms.Compose([
		                          transforms.Resize((opts.resize, opts.resize)),
		                          transforms.ToTensor()]))
	
	LOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=opts.batch_size, shuffle=True)
	

	OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

	WRITER = SummaryWriter(opts.log_path)
	# import ipdb
	# ipdb.set_trace()

	train(loader=LOADER, model=MODEL, optim=OPTIMIZER, writer=WRITER)

