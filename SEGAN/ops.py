# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 02:35:00 2017

@author: Patrick
"""

import numpy as np
import torch
from torch.autograd import Variable

#Padding same function
#
#Parameters:
#   n: input
#   f: filter
#   s: stride
#   o: output_shape
#
#return: input n padded with zeros in shape o
#
def padding_same_1d(n,f,s,o):
    n_shape = list(n.size())
    p_width = ((o-1)*s-n_shape[2]+f)

    if p_width%2 > 0:
        left = int(np.floor(p_width/2.0))
        right = int(np.ceil(p_width/2.0))
    else:
        left = right = int(p_width/2)
    if (type(n) == Variable):
        datatype = n.data.type()
    else:
        datatype = n.type()
    p_left = torch.zeros(n_shape[0],n_shape[1],left).type(new_type=datatype)
    p_right = torch.zeros(n_shape[0],n_shape[1],right).type(new_type=datatype)
#    print(p_left.size())
#    print(p_right.size())
    
    if n.is_cuda:
        p_left = p_left.cuda()
        p_right = p_right.cuda()

    if type(n) == Variable:
        p_left = Variable(p_left)
        p_right = Variable(p_right)

    output = torch.cat((p_left,n,p_right),2)

    return output

def remove_padding_1d(n,o):
    n_shape = list(n.size())
    p_width = n_shape[2] - o

    left = int(np.floor(p_width/2.0))
    right = int(np.ceil(p_width/2.0))

    indices = torch.from_numpy((np.arange(left,n_shape[2]-right))).long()

    if n.is_cuda:
        indices = indices.cuda()

    if type(n) == Variable:
        indices = Variable(indices)

    return torch.index_select(n,2,indices)

##Testing
#x = torch.ones(10,1,16384)
#print(x)
#y = padding_same_1d(x,31,2,8192)
#print(y)
#z = remove_padding_1d(y,16384)
#print(z)
