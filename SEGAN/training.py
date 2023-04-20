# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 02:05:20 2018

@author: Patrick
"""

import SEGAN
import datasets
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import sigproc
import sys, getopt
import datetime
from tensorboardX import SummaryWriter
import librosa

#Default Arguments
epoches = 100
batch_size = 100
loss_fn_dic = {"MSE": nn.MSELoss,"L1": nn.L1Loss,"L2":nn.MSELoss,"CrossEntropy": nn.CrossEntropyLoss}
loss_fn_g = nn.L1Loss
loss_fn_d = nn.MSELoss
optimizer_dic = {"Adam": torch.optim.Adam, "RMSprop": torch.optim.RMSprop, "SGD": torch.optim.SGD}
optimizer = torch.optim.Adam
learning_rate = 0.0001

weight_decay = 0
momentum = 0
cuda = True
input_model = ""


#Get commandline arguments
try:
    opts, args = getopt.getopt(sys.argv[1:],"hc:l:b:e:i:",["help","input-model-path=","cuda=","learning-rate=","batch-size=","epoches=","loss-fn-g=","loss-fn-d=","optimizer=","weight-decay=","momentum="])
except getopt.GetoptError:
    print("training.py [-c,--cuda]")
for opt, arg in opts:
    if opt in ("-h","--help"):
        print("training.py [-i <input-model>,-input-model=] [-c <True,False>,--cuda] [-l <learning-rate>,--learning-rate=] [-b <batch-size>,--batch_size=] [-e <epoches>,--epoches=] [--loss-fn=<L1,MSE,CrossEntropy>] [--optimizer=<Adam,RMSProp,SGD>] [--weight-decay=] [--momentum=]")
        exit()
    if opt in ("-c","--cuda"):
        cuda = True
    if opt in ("-l","--learning-rate"):
        learning_rate = float(arg)
    if opt in ("-b","--batch-size"):
        batch_size = int(arg)
    if opt in ("-e","--epoches"):
        epoches = int(arg)
    if opt in ("--loss-fn-g"):
        if arg not in loss_fn_dic:
            sys.exit("Invalid argument for loss-fn. Choose from: L1,L2,MSE,CrossEntropy")
        loss_fn_g = loss_fn_dic[arg]
    if opt in ("--loss-fn-d"):
        if arg not in loss_fn_dic:
            sys.exit("Invalid argument for loss-fn. Choose from: L1,L2,MSE,CrossEntropy")
        loss_fn_d = loss_fn_dic[arg]
    if opt in ("--optimizer"):
        if arg not in optimizer_dic:
            sys.exit("Invalid argument for optimizer. Choose from: Adam,RMSProp,SGD")
        optimizer = optimizer_dic[arg]
    if opt in ("--weight-decay"):
        weight_decay = float(arg)
    if opt in ("--momentum"):
        momentum = float(arg)

#Create Writer
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
writer = SummaryWriter("./tensorboard/"+date)

#Create datasets
dataset_training = datasets.noisy_clean_28spk_16k_training(cuda=cuda)
dataset_validation = datasets.noisy_clean_28spk_16k_validation(cuda=cuda)

if __name__ == "__main__":
    dataloader_training = torch.utils.data.DataLoader(dataset_training,batch_size=batch_size,num_workers=0,shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation,batch_size=batch_size,num_workers=0,shuffle=True)

#Create model
filter_width = 32
stride = 2
g = SEGAN.Generator(filter_width,stride)
d = SEGAN.Discriminator(filter_width,stride)
if input_model:
    g.load_state_dict(torch.load(input_model))
    d.load_state_dict(torch.load(input_model))
if cuda:
    g.cuda()
    d.cuda()

loss_fn_g = loss_fn_g(size_average=True)
loss_fn_d = loss_fn_d(size_average=True)
if optimizer == torch.optim.Adam:
    g_optimizer = optimizer(g.parameters(),lr=learning_rate,weight_decay=weight_decay)
    d_optimizer = optimizer(g.parameters(),lr=learning_rate,weight_decay=weight_decay)
else:
    g_optimizer = optimizer(g.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum)
    d_optimizer = optimizer(g.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum)

#Training Loop
loss_val_min = 1000.0
val_stop = 0

for e in range(epoches):
    print("epoche",e)
    loss_train = []
    loss_train_adv = []
    loss_val = []


    #Training Loop
    for batch_data,batch_target in dataloader_training:
#        for i, param in enumerate(g.parameters()):
#            writer.add_histogram(str(type(param))+str(i)+".grad", param.grad, e, bins='doane')

        batch_data = Variable(batch_data)
        batch_target = Variable(batch_target)

        #Discriminator training
        #input+real
        d_true = Variable(torch.ones(batch_data.shape[0],1,1))
        d_false = Variable(torch.zeros(batch_data.shape[0],1,1))    
        if cuda:
            d_true = d_true.cuda()
            d_false = d_false.cuda()

        d.zero_grad()
        d_data = Variable(batch_data.data)
        d_target = Variable(batch_target.data)
        d_decision = d(torch.cat([d_data,d_target],1))
        d_real_loss = loss_fn_d(d_decision,d_true)
        d_real_loss.backward()
        print("d real loss:",d_real_loss.data[0])

        #input+fake
        g_fake = g(batch_data)
        d_fake_decision = d(torch.cat([d_data,g_fake.detach()],1))
        d_fake_loss = loss_fn_d(d_fake_decision,d_false)
        d_fake_loss.backward()
        d_optimizer.step()
        print("d fake loss:",d_fake_loss.data[0])


        #Generator training
        g.zero_grad()
        g_decision = d(torch.cat([batch_data,g_fake],1))
        g_loss_adv = loss_fn_d(g_decision,d_true)
        g_loss = loss_fn_g(g_fake,batch_target)
        print("g adv loss:",g_loss_adv.data[0])
        print("g loss:",g_loss.data[0])

        #loss_train = loss_fn(y_pred,batch_target)
        loss_train.append(g_loss.data[0])
        loss_train_adv.append(g_loss_adv.data[0])
#        optimizer.zero_grad()
        #loss_train.backward()
#        g_loss_adv.backward(retain_graph=True)
        g_loss.data[0] += g_loss_adv.data[0]
        g_loss.backward()
        g_optimizer.step()

    loss_train = np.mean(np.asarray(loss_train))
    loss_train_adv = np.mean(np.asarray(loss_train_adv))


    #Validation
    for batch_data,batch_target in dataloader_validation:
        batch_data = Variable(batch_data)
        batch_target = Variable(batch_target)


        y_pred = g(batch_data)
        loss_val.append(loss_fn_g(y_pred,batch_target).data[0])

    loss_val = np.mean(np.asarray(loss_val))


    #Logging for Visualization
#    for i, param in g.parameters():
#        writer.add_histogram(str(type(param))+str(i), param.clone().cpu().data.numpy(), e,bins='doane')
    writer.add_scalars("data/loss",{"training": loss_train,"training adversarial": loss_train_adv,"validation": loss_val},e)


    #Early stopping
    print("validation",loss_val)

    if loss_val < loss_val_min:
        loss_val_min = loss_val
        val_stop = 0
    else:
        val_stop+=1

    if val_stop > 5:
        print("Early stopping")
        break


    #Create wav file for intermediate results
    if e%5 == 0:
        #create audio
        dataset = datasets.noisy_clean_28spk(cuda=cuda)
        wave,target = dataset[50]

        if e == 0:
            wave = wave.cpu().numpy()
            wave = sigproc.deframesig(wave,0,16384,16384)
#            wave = np.trim_zeros(wave,"b")
#            wave = sigproc.deemphasis(wave)
            librosa.output.write_wav("./tensorboard/"+date+'/noisy_input.wav', wave, 16000)
            #writer.add_audio("original",wave,0,sample_rate=16000)
        else:
            wave.resize_(wave.size()[0],1,16384)
            wave = Variable(wave)
            denoised_audio = g(wave)
            denoised_audio = denoised_audio.data
            denoised_audio.resize_(denoised_audio.size()[0],16384)
            denoised_audio = denoised_audio.cpu().numpy()
            denoised_audio = sigproc.deframesig(denoised_audio,0,16384,16384)
#            denoised_audio = np.trim_zeros(denoised_audio,"b")
#            denoised_audio = sigproc.deemphasis(denoised_audio)
    #        writer.add_audio("denoised",denoised_audio.astype(short),e,sample_rate=16000)
            librosa.output.write_wav("./tensorboard/"+date+'/epoche'+str(e)+'.wav', denoised_audio, 16000)
#            writer.add_graph(g, (wave,))




torch.save(g.state_dict(),"./generator_model.pt")
torch.save(d.state_dict(),"./discriminator_model.pt")
writer.export_scalars_to_json("./tensorboard/"+date+"/all_scalars.json")
writer.close()
