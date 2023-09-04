# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:58:41 2021

@author: mitur
"""


import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy

from filterbank_shape import FilterbankShape 
from sklearn.metrics import cohen_kappa_score


class SeqSleepPL(pl.LightningModule):
    def __init__(self,L=1,nChan=1,dropOutProb=0.10,learning_rate=1e-3,weight_decay=1e-3):
        super().__init__()
         
        #save input:
        self.save_hyperparameters()

        #settings:
        self.L=L #sequence length
        self.nChan=nChan
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        
        self.nHidden=64
        self.nFilter=32
        self.attentionSize=64
        self.dropOutProb=dropOutProb
        self.timeBins=29

        #---------------------------filterbank:--------------------------------
        filtershape = FilterbankShape()
        
        #triangular filterbank shape
        shape=torch.tensor(filtershape.lin_tri_filter_shape(nfilt=self.nFilter,
                                                            nfft=256,
                                                            samplerate=100,
                                                            lowfreq=0,
                                                            highfreq=50),dtype=torch.float)
        
        self.Wbl = nn.Parameter(shape,requires_grad=False)
        #filter weights:
        self.Weeg = nn.Parameter(torch.randn(self.nFilter,self.nChan))
        #----------------------------------------------------------------------

        self.epochrnn = nn.GRU(self.nFilter,self.nHidden,1,bidirectional=True,batch_first=True)

        #attention-layer:       
        self.attweight_w  = nn.Parameter(torch.randn(2*self.nHidden, self.attentionSize))
        self.attweight_b  = nn.Parameter(torch.randn(self.attentionSize))
        self.attweight_u  = nn.Parameter(torch.randn(self.attentionSize))
        
        #epoch sequence block:
        self.seqDropout=torch.nn.Dropout(self.dropOutProb, inplace=False)
        self.seqRnn=nn.GRU(self.nHidden*2,self.nHidden,1,bidirectional=True,batch_first=True)
        
        #output:
        self.fc=nn.Linear(2*self.nHidden,5)

    def forward(self, x):
        batch_size, num_epochs, num_sequences, num_features = x.shape
        
        x = torch.reshape(x, (-1, num_sequences, num_features))
        flattener = torch.nn.Unflatten(2, (num_features, 1))
        x = flattener(x)
        
        assert (x.shape[0]/self.L).is_integer()    #we need to pass a multiple of L epochs
        assert (x.shape[1]==self.timeBins)
        assert (x.shape[2]==129)
        assert (x.shape[3]==self.nChan)
        #print("Input shape:")
        #print(x.shape)
        x=x.permute([0,3,1,2])
        
        #print("Swapped channel:")
        #print(x.shape)
        # import pdb; pdb.set_trace()
        #filtering:
                    
        Wfb = torch.multiply(torch.sigmoid(self.Weeg[:,0]),self.Wbl)
        x = torch.matmul(x, Wfb) # filtering
        x = torch.reshape(x, [-1, self.timeBins, self.nFilter])       
        # x=torch.einsum('btrc,ri,ic->btic',x,self.Wbl,torch.sigmoid(self.Weeg))
        # x=torch.reshape(x,(-1,self.timeBins,self.nFilter*self.nChan))
        
        #this uses Einstein notation. letting b:batch, t:time,r:frequency, i:filter, c:channel, 
        #we are saying that the btic'th value in the output should be x_btrc*Wbl_ri*Weeb*ic, summing over 
        #all frequencies. this is a generalization of 
        # Wfb = torch.multiply(Weeg,Wbl)
        # x = torch.matmul(x, Wfb) 
        #which works for n channels        
        
        #biGRU:
        #print("After filter:")
        #print(x.shape)
        x,hn = self.epochrnn(x)
        x=self.seqDropout(x)
        #print("After epoch RNN:")
        #print(x.shape)

        #attention:
        v = torch.tanh(torch.matmul(torch.reshape(x, [-1, self.nHidden*2]), self.attweight_w) + torch.reshape(self.attweight_b, [1, -1]))
        vu = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, self.timeBins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x = torch.sum(x * torch.reshape(alphas, [-1, self.timeBins,1]), 1)
        #print("After attention:")
        #print(x.shape)
        #sequences of epochs:
        x=x.reshape(-1,self.L,self.nHidden*2)   
        #print("Before sequence RNN")
        #print(x.shape)
        x,hn = self.seqRnn(x)
        x=self.seqDropout(x)
        #print("After sequence RNN:")
        #print(x.shape)
        #return to epochs:
        x=x.reshape(-1,self.nHidden*2)
        #print("Before FC")
        #print(x.shape)
        #out:
        x = self.fc(x)
        #print("After FC")
        #print(x.shape)
        return x
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
    
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          mode='max',
                                                          factor=0.5,
                                                          patience=50,
                                                          threshold=0.0001,
                                                          threshold_mode='rel',
                                                          cooldown=0,
                                                          min_lr=0,
                                                          eps=1e-08,
                                                          verbose=True)
        return optimizer
        #return {
        #'optimizer': optimizer,
        #'lr_scheduler': scheduler,
        #'monitor': 'valKappa'
         #}
    
    # def epochMetrics(self,epochOutputs):
    #     epochPreds=[]
    #     trueLabels=[]
    #     totLoss=0
    #     for out in epochOutputs:
    #         epochPreds=np.append(epochPreds,out['pred_labels'].cpu())
    #         trueLabels=np.append(trueLabels,out['labels'].cpu())
    #         totLoss+=out['loss'].cpu()
        
    #     totLoss/=trueLabels.size 
    #     kappa=np.around(cohen_kappa_score(epochPreds,trueLabels),4)
    #     acc=np.mean(epochPreds==trueLabels)
        
    #     return totLoss,kappa,acc
    
    def training_step(self, batch, batch_idx):
        xtemp, ytemp = batch            
        y_pred = self(xtemp)
        
        ytemp = torch.reshape(ytemp, (-1, 5))
        loss = F.cross_entropy(y_pred, ytemp)
        
        a,pred_labels=torch.max(y_pred,1) #b is 0-4
        return loss
        #return {'loss':loss,'pred_labels':pred_labels,'labels':ytemp,'idx':batch_idx }
    
    # def training_epoch_end(self, training_step_outputs):
    #     totLoss,kappa,acc=self.epochMetrics(training_step_outputs)
        
    #     # self.print('training outputs:',totLoss,kappa,acc)
        
    #     self.log('trainLoss',totLoss)
    #     self.log('trainKappa',kappa)

       
    def validation_step(self, batch, batch_idx):
        xtemp, ytemp =batch            
        y_pred = self(xtemp)

        ytemp = torch.reshape(ytemp, (-1, 5))

        loss = F.cross_entropy(y_pred,ytemp)
        a,pred_labels=torch.max(y_pred.cpu(),1) #b is 0-4
        return loss
        #return {'loss':loss,'pred_labels':pred_labels,'labels':ytemp,'idx':batch_idx }
    
    # def validation_epoch_end(self, validation_step_outputs):
    #     totLoss,kappa,acc=self.epochMetrics(validation_step_outputs)
        
    #     # self.print('validation outputs:',totLoss,kappa,acc)
        
    #     self.log('valLoss',totLoss)
    #     self.log('valKappa',kappa)
        
    def test_step(self, batch, batch_idx):
        xtemp, ytemp = batch    
        y_pred = self(xtemp)

        return {'loss':None,'y_pred':y_pred,'idx':batch_idx }
    
    # def test_epoch_end(self, test_step_outputs):
    #     #get dimensions:
    #     nRows=0
    #     for out in test_step_outputs:
    #         nRows+=len(out['idx'])

        
    #     y_pred=np.zeros((nRows,5))
    #     idxs=[]
    #     for out in test_step_outputs:
    #         y_pred[out['idx'].cpu().numpy().astype(int),:]=out['y_pred'].cpu()
    #         idxs=np.append(idxs,out['idx'].cpu())


    #     return {'y_pred':y_pred,'idxs':idxs}
        
    
    # def custom_ensemble_test(self,testX,trainer):
    #         nTest=testX.shape[0]
            
    #         #we need to pad the end to make sure everything fits in N*L epochs
    #         missing=int(np.ceil(nTest/self.L)*self.L-nTest)
            
    #         paddedX=np.concatenate((testX,testX[0:missing,:,:,:]),axis=0)
    #         nPadded=paddedX.shape[0]
            
    #         probs=np.zeros((self.L,nTest,5))
            
    #         paddedX_tensor=torch.tensor(paddedX)
    #         with torch.no_grad():
    #             self.eval()
                
    #             for j in (range(0,self.L)):
    #                 rolledTest=torch.utils.data.TensorDataset(torch.roll(paddedX_tensor,j,0),torch.tensor(range(nPadded)))
    #                 testLoader=torch.utils.data.DataLoader(rolledTest,batch_size=self.L*5,
    #                                                  shuffle=False,drop_last=False)
    #                 epochProbs=np.zeros((nPadded,5))
                    
    #                 testResults=trainer.test(self, testLoader,verbose=False); testResults=testResults[0]
    #                 epochProbs[testResults['idxs'].astype(int),:]=testResults['y_pred']
              
    #                 probs[j,:,:]=scipy.special.softmax(np.roll(epochProbs[0:nTest,:],-j,0),1)
                    
                    
            
    #         probs=probs[:,0:nTest,:]
                
    #         # lprobs=torch.log(torch.tensor(probs))
    #         # y_pred=sum(lprobs,0)
            
    #         #we no longer sum logarithms, but instead sum directly, since Kenneth found that to be better:
    #         y_pred=sum(torch.tensor(probs),0)
            
    #         return {'ensemble_pred':y_pred,'rolled_probs':probs}
   
# raise SystemExit(0)        
            
# import pdb; pdb.set_trace()

# if __name__ == '__main__':
    
#     net=SeqSleepPL(3)
#     x=torch.randn(3*net.L,29,129,1)
#     x[1,:,2,:]=1
#     output=net(x)
#     print('unit test 1 complete')