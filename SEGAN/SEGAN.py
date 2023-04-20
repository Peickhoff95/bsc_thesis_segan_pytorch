# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:53:30 2017

@author: Patrick
"""
#
import torch
from torch.autograd import Variable
from torch import nn

import ops

class Generator(nn.Module):
    def __init__(self,f,s,nfmaps=11):
        super(Generator, self).__init__()
        self.filter = f
        self.stride = s
        self.padding = (f-s)//2
        self.nfmaps = nfmaps
        self.enc_fmaps=[1,16,32,32,64,64,128,128,256,256,512,1024]
        self.dec_fmaps=[2048,1024,512,512,256,256,128,128,64,64,32,1]

        self.encoding = nn.ModuleList()
        self.encoding_prelus = nn.ModuleList()
        self.decoding = nn.ModuleList()
        self.decoding_prelus = nn.ModuleList()

        for i in range(self.nfmaps):
            self.encoding.append(nn.Conv1d(self.enc_fmaps[i],self.enc_fmaps[i+1],self.filter,self.stride,padding=self.padding))
            self.encoding_prelus.append(nn.PReLU())
            self.decoding.append(nn.ConvTranspose1d(self.dec_fmaps[i],self.enc_fmaps[-i-2],self.filter,self.stride,padding=self.padding))
            if (i < self.nfmaps-1):
                self.decoding_prelus.append(nn.PReLU())

    def forward(self, x):
        skip = []
        for i in range(self.nfmaps):
            x = self.encoding[i](x)
            x = self.encoding_prelus[i](x)
            if (i < self.nfmaps-1):
                skip.append(x)

        z = Variable(torch.normal(torch.ones(x.size()))).cuda()
        x = torch.cat((x,z),1)

        for i in range(self.nfmaps):
            x = self.decoding[i](x)
            if (i < self.nfmaps-1):
                x = self.decoding_prelus[i](x)
                x = torch.cat((x,skip[-i-1]),1)

        return x

class Discriminator(nn.Module):
    def __init__(self,f,s,nfmaps=11):
        super(Discriminator, self).__init__()
        self.filter = f
        self.stride = s
        self.padding = (f-s)//2
        self.fmaps = [2,16,32,32,64,64,128,128,256,256,512,1024]
        self.layers = nn.ModuleList()

        for i in range(nfmaps):
            self.layers.append(nn.Conv1d(self.fmaps[i],self.fmaps[i+1],self.filter,self.stride,padding=self.padding))
            self.layers.append(nn.BatchNorm1d(self.fmaps[i+1]))
            self.layers.append(nn.LeakyReLU(0.3))

        self.layers.append(nn.Conv1d(self.fmaps[nfmaps],1,1,1))
        self.layers.append(nn.Linear(16384//(2**nfmaps),1))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

#filter_width = 32
#stride = 2
#####
#i = Variable(torch.randn(1,1,16384))
#r = Variable(torch.ones(1,1,16384))
#i = i.cuda()
#r.cuda()
#g = Generator(filter_width,stride)
#d = Discriminator(filter_width,stride)
#g.cuda()
#d.cuda()
#y = d.forward(i)
#print(y)
#
#loss_fn = nn.MSELoss(size_average=True)
#learning_rate = 0.0001
#g_optim = torch.optim.Adam(g.parameters(),lr=learning_rate)
#d_optim = torch.optim.Adam(d.parameters(),lr=learning_rate)
#
#d_true = Variable(torch.ones(1,1,1))
#d_false = Variable(torch.zeros(1,1,1))
#for e in range(1000):
#    print(e)

#Discriminator training
    #input+real
#    j = Variable(i.data)
#    d_decision = d(torch.cat([j,r],1))
#    d_real_loss = loss_fn(d_decision,d_true)
#    d_real_loss.backward()
#    print("d real loss:",d_real_loss.data[0])

    #input+fake
#    fake = g(i).detach()
#    d_fake_decision = d(torch.cat([i,fake],1))
#    d_fake_loss = loss_fn(d_fake_decision,d_false)
#    d_fake_loss.backward()
#    d_optim.step()
#    print("d fake loss:",d_fake_loss.data[0])

#Generator training

#    g.zero_grad()

#    g_fake = g(i)
#    d_decision = d(torch.cat([i,g_fake],1))
#    g_loss = loss_fn(d_decision,d_true)
#    g_loss.backward()
#    g_optim.step()
#    print("g loss:",g_loss.data[0])
#    print("g loss to real:",loss_fn(g_fake,r).data[0])
