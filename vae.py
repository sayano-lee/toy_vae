from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


class Encoder(nn.Module):

    def __init__(self, nc, ndf, nlatent):
        # nc  = 3  color image by default
        # ndf = 64 encoder intermediate variable dimension
        # nlatent = 1024 extracted latent variable dimension
        
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)
    
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf)
    
        self.conv3 = nn.Conv2d(ndf, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*4)
        
        self.conv5 = nn.Conv2d(ndf*4, ndf*16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf*16)

        self.conv6 = nn.Conv2d(ndf*16, ndf*16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(ndf*16)
        
        # last convolutional layer

        self.conv7 = nn.Conv2d(ndf*16, nlatent, kernel_size=4, stride=1, padding=0, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.max_pool2x2 = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        # print('original size {}'.format(x.shape))

        y1 = self.lrelu(self.conv1(x))            #(bn, nc, in_size/2, in_size/2)
        # print('conv1 shape {}'.format(y1.shape))

        y2 = self.lrelu(self.bn2(self.conv2(y1))) #(bn, nc, in_size/4, in_size/4)
        y2 = y2 + self.max_pool2x2(y1)
        # print('conv2 shape {}'.format(y2.shape))

        y3 = self.lrelu(self.bn3(self.conv3(y2))) #(bn, nc, in_size/8, in_size/8)
        # y3 = y3 + self.max_pool2x2(y2)
        # print('conv3 shape {}'.format(y3.shape))

        y4 = self.lrelu(self.bn4(self.conv4(y3))) #(bn, nc, in_size/16, in_size/16)
        y4 = y4 + self.max_pool2x2(y3)
        # print('conv4 shape {}'.format(y4.shape))

        y5 = self.lrelu(self.bn5(self.conv5(y4))) #(bn, nc, in_size/32, in_size/32)
        # y5 = y5 + self.max_pool2x2(y4)
        # print('conv5 shape {}'.format(y5.shape))

        y6 = self.lrelu(self.bn6(self.conv6(y5))) #(bn, nc, in_size/64, in_size/64)
        # print('conv6 shape {}'.format(y6.shape))

        y7 = self.conv7(y6)
        
        # out = self.sigmoid(y7)

        out = y7
        
        return out


class Decoder(nn.Module):

    def __init__(self, nc, ngf, nlatent):
        # nc  = 3  color image by default
        # ndf = 64 encoder intermediate variable dimension
        # nlatent = 1024 extracted latent variable dimension

        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(nlatent, ngf*16, kernel_size=4, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(ngf*16)
        
        self.deconv2 = nn.ConvTranspose2d(ngf*16, ngf*16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*16)
        
        self.deconv3 = nn.ConvTranspose2d(ngf*16, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        
        self.deconv4 = nn.ConvTranspose2d(ngf*4, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf*4)

        self.deconv5 = nn.ConvTranspose2d(ngf*4, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        
        self.deconv6 = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        
        self.deconv7 = nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
 
        y1 = self.relu(self.bn1(self.deconv1(x)))
        
        y2 = self.relu(self.bn2(self.deconv2(y1)))
        
        y3 = self.relu(self.bn3(self.deconv3(y2)))
        
        y4 = self.relu(self.bn4(self.deconv4(y3)))
        
        y5 = self.relu(self.bn5(self.deconv5(y4)))
        
        y6 = self.relu(self.bn6(self.deconv6(y5)))
        
        y7 = self.deconv7(y6)

        # out = self.sigmoid(y7)
        out = y7

        return out


class VAE(nn.Module):
    def __init__(self, nz, nc, ndf, ngf, nlatent):
        super(VAE, self).__init__()

        self.have_cuda = False
        self.nz = nz
        self.nlatent = nlatent

        self.encoder = Encoder(nc, ndf, nlatent)
        self.decoder = Decoder(nc, ngf, nlatent)

        self.fc1 = nn.Linear(nlatent, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, nlatent)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # h1 = self.fc1(conv.view(-1, self.nlatent))
        # return self.fc21(h1), self.fc22(h1)
        return conv

    def decode(self, z):
        # h3 = self.relu(self.fc3(z))
        # deconv_input = self.fc4(h3)
        # deconv_input = deconv_input.view(-1, self.nlatent, 1, 1)
        decoded = self.decoder(z)
        return decoded

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        feat = self.encode(x)
        decoded = self.decode(feat)
        return decoded
