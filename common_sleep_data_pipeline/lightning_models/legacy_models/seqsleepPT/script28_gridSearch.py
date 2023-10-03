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
tempMat='D:\\OneDrive - Aarhus Universitet\\python\\data\\20x4_nights\\mat\\'
if  not found and os.path.exists(tempMat):
    matDir=tempMat
    found=True    
    print(matDir)

assert(found)

#comes pre-normalized:
loadedData=sleepEEGcontainer1.fromDirectory(matDir,deriv='eeg_lr')
print('Data loaded')


#%%

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
    #%% setup neptune
    
neptune_token=os.getenv('NEPTUNE_API_TOKEN')
#documentation: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/neptune.py
neptune_logger = NeptuneLogger(
    api_key=neptune_token,
              project_name='mikkelsen.kaare/seqsleep-pytorch',
              params=params,
              experiment_name=expName,
              close_after_fit=False,
              tags=['filterbank','lossStop'])
    

#%% loop over test sets & train



pl.seed_everything(seed)
allKappas=np.zeros(20)

for iTest in range(20):
    #testSet:
    testIndxs=np.array([iTest])
    print('testIndxs:',testIndxs)
    
    #train, validation
    rest=np.delete(np.array(range(1,21)),testIndxs-1)
    assert len(rest)==19
    tempOrder=np.random.permutation(len(rest))
    tempOrder=rest[tempOrder]
    trainIndxs=tempOrder[0:15]
    valIndxs=tempOrder[15:19]
    
    #load data
    trainX,trainy,trainLabels=loadedData.returnBySubject(trainIndxs)
    valX,valy,valLabels=loadedData.returnBySubject(valIndxs)
    
    trainLabels_tensor=torch.tensor(trainLabels-1).type(torch.long)
    valLabels_tensor=torch.tensor(valLabels-1).type(torch.long)
    
    #pytorch datasets:
    trainDataset=torch.utils.data.TensorDataset(torch.tensor(trainX),torch.squeeze(trainLabels_tensor),torch.tensor(range(trainLabels.size)))
    valDataset=torch.utils.data.TensorDataset(torch.tensor(valX),torch.squeeze(valLabels_tensor),torch.tensor(range(valLabels.size)))
    
    #dataLoaders:
    trainSampler=torch.utils.data.DataLoader(trainingEEGDataset_1(trainDataset,L),batch_size=1,
                                             shuffle=True,drop_last=True,collate_fn=custom_collate_fn)
    valSampler=torch.utils.data.DataLoader(valDataset,batch_size=L*5,
                                             shuffle=False,drop_last=True)
    
    print("make a clean net:")
    net=SeqSleepPL(L,1,dropOutProb,learning_rate,weight_decay)    
    
    #start training:
    early_stopping = pl.callbacks.EarlyStopping(
       monitor='valKappa',
       min_delta=0.00,
       patience=earlyStoppingDelay,
       verbose=True,
       mode='max'
    )
    
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    
    trainer = pl.Trainer(max_epochs=1500,deterministic=True,gpus=1,
                         callbacks=[early_stopping,lr_monitor],
                         logger=neptune_logger,
                         # fast_dev_run=2)
                          benchmark=True, #speeds up training if batch size is constant
                          progress_bar_refresh_rate=0
                          )
    
    trainer.fit(net,trainSampler,valSampler)
    
    
     #%% testing
    testX,testy,testLabels=loadedData.returnBySubject(testIndxs)
    
    ensembleTesting=net.custom_ensemble_test(testX,trainer)
    a,b=torch.max(ensembleTesting['ensemble_pred'].cpu().clone().detach(),1) #b is 0-4
    kappa=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)
    
    rolledKappas=np.zeros(L)
    for iRoll in range(L):
        a,b=torch.max(torch.tensor(ensembleTesting['rolled_probs'][iRoll,:,:]),1) #b is 0-4   
        rolledKappas[iRoll]=cohen_kappa_score(torch.unsqueeze(b+1,1),testLabels.T)
        
    print('rolledKappas:',rolledKappas)
    print('meanRolledKappa:',np.mean(rolledKappas))
    print('Consensus:',testIndxs,kappa)
    neptune_logger.log_metric('subjectKappa',kappa)
    neptune_logger.log_metric('meanRolledKappa',np.mean(rolledKappas))
    
    allKappas[testIndxs-1]=kappa


np.savetxt(expName + '_allKappas.csv', allKappas, delimiter=",")

print('allKappas',allKappas)
print('meanKappas',np.mean(allKappas))
neptune_logger.log_artifact(expName + '_allKappas.csv')
neptune_logger.log_metric('meanKappa',np.mean(allKappas))

neptune_logger.close()
#%%

