# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:46:50 2021

@author: mitur
"""

import numpy as np
import h5py
import os
import pickle
from pathlib import Path
import torch

class sleepEEGcontainer1:
    def __init__(self,inputDict):
        self.Xlist=inputDict['Xlist']
        self.ylist=inputDict['ylist']
        self.labelList=inputDict['labelList']
        self.subjectName=inputDict['subjectName']
        self.subjectNight=inputDict['subjectNight']
        
        self.n=len(self.Xlist)
        
        self.normalize()

    def __repr__(self):
        return 'Dataset with ' + str(self.n) + ' recordings'
    
    def normalize(self):
        self.n=len(self.Xlist)
        assert self.n == len(self.ylist)
        assert self.n == len(self.labelList)
        
        #normalize data (for each frequency):
        allMeans=np.array([np.mean(x,axis=(1,2)) for x in self.Xlist if len(x)>10])
        totMean=np.mean(allMeans,0).reshape((-1,1,1))
        Xlist=[x-totMean for x in self.Xlist]
        
        allStds=np.array([np.std(x,axis=(1,2)) for x in self.Xlist if len(x)>10])
        totStd=np.mean(allStds,0).reshape((-1,1,1))
        self.Xlist=[x/totStd for x in Xlist]
        
    
    
    @classmethod
    def fromDirectory(cls,matDir,deriv):
        loadedDict=loadMatData(matDir,deriv)
        return cls(loadedDict)

        
    def returnRecords(self,idxs):
        idxs=idxs[0]
        assert np.array(idxs).size

        #ignore empty idxs:
        idxs=[idxs[i] for i in range(len(idxs)) if self.Xlist[idxs[i]].size>1000]
        assert np.array(idxs).size

        Xout=np.array(self.Xlist[idxs[0]])
        yout=np.array(self.ylist[idxs[0]])
        label_out=np.array(self.labelList[idxs[0]])
        

        for i in idxs[1:]:
            Xout=np.concatenate([Xout,self.Xlist[i]],axis=2)
            yout=np.concatenate([yout,self.ylist[i]],axis=1)  
            label_out=np.concatenate([label_out,self.labelList[i]],axis=1)
            
        #we want batch x 29 x 129 x 1:
        Xout=Xout.swapaxes(0,2) 
        Xout=np.expand_dims(Xout,3)
        
        #we want batch x 5:
        yout=yout.T
        
        return Xout,yout,label_out
    
    def returnBySubject(self,iSs):
        assert np.array(iSs).size
        
        #did the user ask for non-existent subjects:
        recs=np.in1d(iSs,self.subjectName)
        if not all(np.in1d(iSs,self.subjectName)):
            print('Error: requested subject not in data set')
            raise SystemExit(0)    
        
        #find recordings for all subjects:
        recs=np.where(np.in1d(self.subjectName,iSs))
   
        Xout,yout,label_out=self.returnRecords(recs)
        return Xout,yout,label_out

        
    
def loadMatData(matDir,deriv):
    pickleName=os.path.join(matDir,deriv+'_'+'pickled.p')
    print('Pickle-name:',pickleName)
    if os.path.exists(pickleName):
        print('Loading pickled data')
        temp=pickle.load(open(pickleName,'rb'))
        Xlist=temp['Xlist']
        ylist=temp['ylist']
        labelList=temp['labelList']
        subjectName=temp['subjectName']
        subjectNight=temp['subjectNight']
        
    else:    
        Xlist=[0] #dummy first value
        ylist=[0]
        labelList=[0]
        subjectName=np.empty((0,))
        subjectNight=np.empty((0,))
        counter=0
        #get subject-dirs:
        p = Path(matDir)
        subjectDirs=[x for x in p.iterdir() if x.is_dir()]

        for iS in range(len(subjectDirs)):
            try:
                subjectName_temp=int(str(subjectDirs[iS])[-2:])
            except:
                subjectName_temp=int(iS)
            
            #get night-dirs:
            p =subjectDirs[iS]
            nightDirs=[x for x in p.iterdir() if x.is_dir()]
            for iN in range(len(nightDirs)):
                
                filename = os.path.join(nightDirs[iN], deriv+'.mat')
                temp=h5py.File(filename,'r')
                subjectName=np.append(subjectName,subjectName_temp)
                subjectNight=np.append(subjectNight,iN)
                Xlist+=[np.array(temp['X'])]
                try:
                    ylist+=[np.array(temp['y'])]        
                    labelList+=[np.array(temp['label'])]
                except:
                    #if there are no labels:
                    ylist+=[np.array(np.empty((0,0)))]        
                    labelList+=[np.empty((0,0))]
                
                counter+=1
        
        Xlist.pop(0)
        ylist.pop(0)
        labelList.pop(0)
    
        print('Pickling data')
        pickle.dump({'Xlist':Xlist,'ylist':ylist,'labelList':labelList,'subjectName':subjectName,'subjectNight':subjectNight}, open( pickleName, "wb" ) )
    
    return {'Xlist':Xlist,'ylist':ylist,'labelList':labelList,'subjectName':subjectName,'subjectNight':subjectNight}

if __name__ == '__main__':
    matDir='O:\\ST_NTLab\\People\\kaare\\Sleep\\boerneInkon\\temp\\mat_renamed'

    deriv='m2m1'
    loadedData=sleepEEGcontainer1.fromDirectory(matDir,deriv)
    x,y,labels=loadedData.returnBySubject([1])
 


class trainingEEGDataset_1(torch.utils.data.Dataset):
    #a wrapper for torch datasets, to make it possible to shuffle sequences
    def __init__(self, inputDataset:torch.utils.data.Dataset=None,L:int=None):
        
        self.dataSet=inputDataset
        self.L=L
        
        #bookkeeping idx's:
        self.seqIndices=None
        self.getCounter=0
        self.reset()
        
    def reset(self):
        #reset bookkeeping idx's
        start=np.random.randint(0,self.L)
        seqRange=range(start,len(self.dataSet),self.L)
        seqRange=range(seqRange[0],seqRange[len(seqRange)-1])
        self.seqIndices=np.reshape(seqRange,(-1,self.L))
        self.getCounter=0        

        
    def __len__(self):
        return int(np.floor(len(self.dataSet)/self.L)) 
        # return self.seqIndices.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if type(idx) in (tuple,list):
            print(len(idx))
            idx=idx[0]

            
        try:
            self.getCounter+=len(idx)
        except:
            #if idx is a scalar, the other one fails
            self.getCounter+=np.array(idx).size
                
        #because __len__ fluctuates, we need to make sure we don't try to access non-existing data:
        idx=idx%(self.seqIndices.shape[0])

        try:
            sample=self.dataSet[np.reshape(self.seqIndices[idx,:],(-1,))]
        except:
            print('Custom dataloader failed')
            print('maxIdx',np.max(idx))
            print('len:',self.len)
            print('self.dataSet',self.dataSet.shape)
            print('seqIndices.max',np.max(self.seqIndices))
            raise SystemExit(0) 
            
        #if all idxs have been passed:
        if self.getCounter>=(self.seqIndices.shape[0]-1):
            self.reset()

        return sample




def custom_collate_fn(batch):
    x = torch.cat([item[0] for item in batch])
    y = torch.cat([item[1] for item in batch])
    i = torch.cat([item[2] for item in batch])
    return x, y, i
