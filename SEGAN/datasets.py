# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:00:58 2017

@author: Patrick
"""
import torch
import torch.utils.data
import numpy as np
import scipy.io.wavfile
import os
import sigproc

class noisy_clean_28spk(torch.utils.data.Dataset):
    def __init__(self,path_data="../datasets/DS_10283_2791/noisy_trainset_28spk_wav/noisy_trainset_28spk_16kHz_wav",path_target="../datasets/DS_10283_2791/clean_trainset_28spk_wav/clean_trainset_28spk_16kHz_wav",cached=False,cuda=False):

        self.path_data = path_data
        self.path_target = path_target
        self.validation_set = []
        self.training_set = []
        self.cached = cached
        self.cuda = cuda

        self.training_set = (np.asarray(os.listdir(path_data+"/training")))
        self.validation_set = (np.asarray(os.listdir(path_data+"/validation")))

        if self.cached:
            training_data = [[],[]]
            validation_data = []
            for filename in self.training_set:
                data,target = self.loadFrames(filename,train_set="training")
                training_data[0].append(data)
                training_data[1].append(target)
            print(training_data)
            training_data[0] = torch.cat(training_data[0],0)
            training_data[1] = torch.cat(training_data[1],0)
            print(training_data)
            for filename in self.validation_set:
                data,target = self.loadFrames(filename,train_set="validation")
                validation_data[0].append(data)
                validation_data[1].append(target)

            self.training_set = training_data
            self.validation_set = validation_data

        super(noisy_clean_28spk,self).__init__()

    def __getitem__(self,index):
        if self.cached:
            return self.training_set[index]
        return self.loadFrames(self.training_set[index])

    def __len__(self):
        return len(self.training_set)

    def lenValidationSet(self):
        return len(self.validation_set)

    def lenTrainingSet(self):
        return len(self.training_set)

    def getTrainingItem(self,index):
        if self.cached:
            return self.training_set[index]
        return self.loadFrames(self.training_set[index],train_set="training")

    def getValidationItem(self,index):
        if self.cached:
            return self.validation_set[index]
        return self.loadFrames(self.validation_set[index],"validation")

    def loadFrames(self,filename,train_set="training"):
        data = scipy.io.wavfile.read(self.path_data+"/"+train_set+"/"+filename)[1]
        target = scipy.io.wavfile.read(self.path_target+"/"+train_set+"/"+filename)[1]
        
#        data = sigproc.preemphasis(data)
#        target = sigproc.preemphasis(target)
        
        data = sigproc.framesig(data, 16384, 16384)
        target = sigproc.framesig(target, 16384, 16384) 

        vec_fun = np.vectorize((lambda x:x/2**15))
        data_frame_tensor = torch.from_numpy(vec_fun(data)).float()
        target_frame_tensor = torch.from_numpy(vec_fun(target)).float()
        

        if self.cuda:
            data_frame_tensor = data_frame_tensor.cuda()
            target_frame_tensor = target_frame_tensor.cuda()
        
        return data_frame_tensor,target_frame_tensor
    
class noisy_clean_28spk_16k_training(torch.utils.data.Dataset):
    def __init__(self,path_data="../datasets/DS_10283_2791/noisy_trainset_28spk_wav/noisy_trainset_28spk_16kHz_wav/",path_target="../datasets/DS_10283_2791/clean_trainset_28spk_wav/clean_trainset_28spk_16kHz_wav/",cuda=False):

        self.path_data = path_data
        self.path_target = path_target
        self.training_set = []
        self.cuda = cuda

        self.training_set = (np.asarray(os.listdir(path_data+"/training/seconds")))

        super(noisy_clean_28spk_16k_training,self).__init__()

    def __getitem__(self,index):
        return self.loadFrames(self.training_set[index])

    def __len__(self):
        return len(self.training_set)
#        return 1000

    def loadFrames(self,filename):
        data = scipy.io.wavfile.read(self.path_data+"training/seconds/"+filename)[1]
        target = scipy.io.wavfile.read(self.path_target+"training/seconds/"+filename)[1]
        
#        data = sigproc.preemphasis(data)
#        target = sigproc.preemphasis(target)
        
        data = sigproc.framesig(data, 16384, 16384)
        target = sigproc.framesig(target, 16384, 16384) 

        vec_fun = np.vectorize((lambda x:x/2**15))
        data_frame_tensor = torch.from_numpy(vec_fun(data)).float()
        target_frame_tensor = torch.from_numpy(vec_fun(target)).float()
        

        if self.cuda:
            data_frame_tensor = data_frame_tensor.cuda()
            target_frame_tensor = target_frame_tensor.cuda()
        
        return data_frame_tensor,target_frame_tensor

class noisy_clean_28spk_16k_validation(torch.utils.data.Dataset):
    def __init__(self,path_data="../datasets/DS_10283_2791/noisy_trainset_28spk_wav/noisy_trainset_28spk_16kHz_wav/",path_target="../datasets/DS_10283_2791/clean_trainset_28spk_wav/clean_trainset_28spk_16kHz_wav/",cuda=False):

        self.path_data = path_data
        self.path_target = path_target
        self.validation_set = []
        self.cuda = cuda

        self.validation_set = (np.asarray(os.listdir(path_data+"/validation/seconds")))

        super(noisy_clean_28spk_16k_validation,self).__init__()

    def __getitem__(self,index):
        return self.loadFrames(self.validation_set[index])

    def __len__(self):
        return len(self.validation_set)
#        return 1000

    def loadFrames(self,filename):
        data = scipy.io.wavfile.read(self.path_data+"validation/seconds/"+filename)[1]
        target = scipy.io.wavfile.read(self.path_target+"validation/seconds/"+filename)[1]
        
#        data = sigproc.preemphasis(data)
#        target = sigproc.preemphasis(target)
        
        data = sigproc.framesig(data, 16384, 16384)
        target = sigproc.framesig(target, 16384, 16384) 

        vec_fun = np.vectorize((lambda x:x/2**15))
        data_frame_tensor = torch.from_numpy(vec_fun(data)).float()
        target_frame_tensor = torch.from_numpy(vec_fun(target)).float()
        

        if self.cuda:
            data_frame_tensor = data_frame_tensor.cuda()
            target_frame_tensor = target_frame_tensor.cuda()
        
        return data_frame_tensor,target_frame_tensor

class gauss_noise_training(torch.utils.data.Dataset):
    def __init__(self,path_data="../datasets/DS_10283_2791/clean_trainset_28spk_wav/clean_trainset_28spk_16kHz_wav/",cuda=False):

        self.path_data = path_data
        self.training_set = []
        self.cuda = cuda

        self.training_set = (np.asarray(os.listdir(path_data+"/training/seconds")))

        super(gauss_noise_training,self).__init__()

    def __getitem__(self,index):
        return self.loadFrames(self.training_set[index])

    def __len__(self):
#        return len(self.training_set)
        return 2000

    def loadFrames(self,filename):
        data = scipy.io.wavfile.read(self.path_data+"training/seconds/"+filename)[1]

        data = sigproc.framesig(data, 16384, 16384)
        vec_fun = np.vectorize((lambda x:x/2**15))
        data_frame_tensor = torch.from_numpy(vec_fun(data)).float()

        noise = torch.normal(torch.zeros(1,1,16384),0.333)
        target_frame_tensor = torch.add(data_frame_tensor,noise)

        if self.cuda:
            data_frame_tensor = data_frame_tensor.cuda()
            target_frame_tensor = target_frame_tensor.cuda()

        return data_frame_tensor,target_frame_tensor
    
class gauss_noise_validation(torch.utils.data.Dataset):
    def __init__(self,path_data="../datasets/DS_10283_2791/clean_trainset_28spk_wav/clean_trainset_28spk_16kHz_wav/",cuda=False):

        self.path_data = path_data
        self.training_set = []
        self.cuda = cuda

        self.validation_set = (np.asarray(os.listdir(path_data+"/validation/seconds")))

        super(gauss_noise_validation,self).__init__()

    def __getitem__(self,index):
        return self.loadFrames(self.validation_set[index])

    def __len__(self):
      # return len(self.validation_set)
        return 2000

    def loadFrames(self,filename):
        target = scipy.io.wavfile.read(self.path_data+"validation/seconds/"+filename)[1]

        target = sigproc.framesig(target, 16384, 16384)
        vec_fun = np.vectorize((lambda x:x/2**15))
        target_frame_tensor = torch.from_numpy(vec_fun(target)).float()

        noise = torch.normal(torch.zeros(1,1,16384),0.333)
        data_frame_tensor = torch.add(target_frame_tensor,noise)

        if self.cuda:
            data_frame_tensor = data_frame_tensor.cuda()
            target_frame_tensor = target_frame_tensor.cuda()

        return data_frame_tensor,target_frame_tensor


#dataset = noisy_clean_28spk_16k_training(cuda=True)
##print(dataset[0])
##print(dataset[0][1])
#if __name__ == "__main__":
#    dataloader = torch.utils.data.DataLoader(dataset,batch_size=20,num_workers=0)
#    print(next(iter(dataloader)))
##print(dataset.__len__())
##print(dataset.__getitem__(0)[0][0])
##print(dataset.__getitem__(0)[0])
