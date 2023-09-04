# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from sklearn.metrics import cohen_kappa_score
import time


import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger


import scipy.special



#%% check gpu availability

cuda=torch.device('cpu')

if torch.cuda.is_available():
    cuda=torch.device('cuda:0')
    print(torch.version.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f/1e6)
else:
    print("no cuda available")



#%% lightning setup & import seqsleepnet

from seqSleep_pytorchLightning import SeqSleepPL
#    def __init__(self,L=1,nChan=1,dropOutProb=0.10,learning_rate=1e-3,weight_decay=1e-3)



#%% import data

from loadMat5 import sleepEEGcontainer1,trainingEEGDataset_1,custom_collate_fn

#figure out where it is:
found=False

#office:
tempMat='F:\\OneDrive - Aarhus Universitet\\python\\data\\20x4_nights\\mat\\'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True
    print(matDir)

#prime:
tempMat='/com/ecent/NOBACKUP/sleepData/20x4_nights/mat'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True
    print(matDir)
    
#laptop:
tempMat='C:\\Users\\au207178\\OneDrive - Aarhus Universitet\\python\\data\\20x4_nights\\mat\\'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True    
    print(matDir)

assert(found)

#comes pre-normalized:
loadedData=sleepEEGcontainer1.fromDirectory(matDir,deriv='eeg_lr')
print('Data loaded')


#%% parsing commandline input (optional)

L=20
learning_rate=1e-4
seed=4
weight_decay=0.001
dropOutProb=0.01
earlyStoppingDelay=800


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--L", help="sequence length (20)",type=int)
parser.add_argument("--lr", help="learning rate (0.0001)",type=float)
parser.add_argument("--wd", help="weight decay (0.001)",type=float)
parser.add_argument("--dop", help="drop out probability (0.5)",type=float)
parser.add_argument("--seed", help="random seed (3)",type=int)
parser.add_argument("--earlyStop", help="early stopping delay (50)",type=int)

args=parser.parse_args()

if args.L is not None:
    L=args.L
if args.lr is not None:
    learning_rate=args.lr
if args.wd is not None:
    weight_decay=args.wd
if args.dop is not None:
    dropOutProb=args.dop
if args.earlyStop is not None:
    earlyStoppingDelay=args.earlyStop
if args.seed is not None:
    seed=args.seed

expName='gridSearch'
params = {'L': L,
      'learning_rate': learning_rate,
      'seed': seed,
      'weight_decay': weight_decay,
      'dropOutProb': dropOutProb,
      'expName': expName,
      'earlyStoppingDelay':earlyStoppingDelay
      }




trainer = pl.Trainer()
    

#%% load pretrained model
net=SeqSleepPL(L,1,dropOutProb,learning_rate,weight_decay)    

net.load_state_dict(torch.load('image1.mod'))


    
 #%% testing
 
testIndxs=2
testX,testy,testLabels=loadedData.returnBySubject(testIndxs)
# ensembleTesting=net.custom_ensemble_test(testX,trainer)


#temporary fix because the original implementation breaks when newest version of pytorch lightning is used:
nTest=testX.shape[0]
  
#we need to pad the end to make sure everything fits in N*L epochs
missing=int(np.ceil(nTest/net.L)*net.L-nTest)

paddedX=np.concatenate((testX,testX[0:missing,:,:,:]),axis=0)
nPadded=paddedX.shape[0]

probs=np.zeros((net.L,nTest,5))

paddedX_tensor=torch.tensor(paddedX)
with torch.no_grad():
    net.eval()
  
    for j in (range(0,net.L)):
        rolledTest=torch.utils.data.TensorDataset(torch.roll(paddedX_tensor,j,0),torch.tensor(range(nPadded)))
        testLoader=torch.utils.data.DataLoader(rolledTest,batch_size=net.L*5,
                                         shuffle=False,drop_last=False)
        epochProbs=np.zeros((nPadded,5))
        
        for paddedX,idxs in testLoader:
            testResults=net(paddedX)
            epochProbs[idxs.numpy().astype(int),:]=testResults
  
        probs[j,:,:]=scipy.special.softmax(np.roll(epochProbs[0:nTest,:],-j,0),1)

y_pred=sum(torch.tensor(probs),0)


a,b=torch.max(y_pred.cpu().clone().detach(),1) #b is 0-4
kappa=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)

rolledKappas=np.zeros(L)
for iRoll in range(L):
    a,b=torch.max(torch.tensor(probs[iRoll,:,:]),1) #b is 0-4   
    rolledKappas[iRoll]=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)
    
print('rolledKappas:',rolledKappas)
print('meanRolledKappa:',np.mean(rolledKappas))
print('Consensus:',kappa)



#%%

