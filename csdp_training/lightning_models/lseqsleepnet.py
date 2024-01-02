# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch
from csdp_training.utility import acc, kappa, f1, log_test_step
import math
from csdp_training.lightning_models.base import Base_Lightning

class LSeqSleepNet_Lightning(Base_Lightning):
    def __init__(self, 
                 lseqsleep,
                 lr,
                 batch_size,
                 lr_patience,
                 lr_factor,
                 lr_minimum):
        
        super().__init__(lseqsleep, 
                         lr, 
                         batch_size,
                         lr_patience,
                         lr_factor,
                         lr_minimum)
    
    def prep_training_batch(self, x, y):        
        assert x.dim() == 5
        assert y.dim() == 2
        
        y = torch.flatten(y, 0, 1)
        x = x.float()

        # From (batch_size, num_channels, 200, 29, 129) --> (batch_size, 200, num_channels, 29, 129)
        x = x.swapaxes(1,2)

        assert(x.dim() == 5)
        assert(y.dim() == 1)

        return x, y
    
    def compute_epoch_metrics(self, epoch_outputs):
        preds = []
        labels = []
        totLoss = 0
        
        for step in epoch_outputs:
            loss = step['loss']
            pred = step['y_pred']
            label = step['labels']
            
            totLoss+=loss
            preds.append(pred)
            labels.append(label)
        
        preds = torch.cat(preds)
        labels = torch.cat(labels)

        a = acc(preds, labels)
        k = kappa(preds, labels)
        f = f1(preds,labels, average=False)
        
        return totLoss, a, k, f
    
    def training_step(self, batch, _):
        x_eegs, x_eogs, y_temp, _ = batch

        x_temp = torch.cat([x_eegs, x_eogs], dim=1)
  
        assert x_temp.shape[1] == 2
        
        x_temp, y_temp = self.prep_training_batch(x_temp, y_temp)
       
        y_pred = self(x_temp)
        y_pred = torch.reshape(y_pred, (-1, 5))
        loss = self.loss(y_pred, y_temp.long())   

        self.training_step_outputs.append(loss)

        return loss
       
    def predict_single_channel(self, x_eegs, x_eogs, y_temp: torch.Tensor):
        # Assumes x_eegs, x_eogs to be: (Channels, Epochs, 29, 129)
        # Assumes y_temp to be 1D --> (Number of epochs)

        x_eeg = x_eegs[0,...]
        x_eog = x_eogs[0,...]

        x_temp = torch.stack([x_eeg, x_eog], dim=0)
        
        assert x_temp.shape[0] == 2
        x_temp = torch.swapaxes(x_temp, 0, 1)
        
        assert (x_temp.shape[0] % 200) == 0 
        
        x_temp = torch.reshape(x_temp, (-1, 200, 2, 29, 129))
  
        y_pred = self(x_temp.float())
        y_pred = torch.reshape(y_pred, (-1, 5))

        loss = self.loss(y_pred, y_temp.long()) 
        
        y_pred = torch.argmax(y_pred, dim=1)
        
        return y_pred, loss
    
    def validation_step(self, batch, _):
        x_eeg, x_eog, y_temp, _ = batch

        if x_eog.shape[1] == 0:
            print("Found no EOG channel, duplicating EEG instead")
            x_eog = x_eeg

        x_eeg = torch.squeeze(x_eeg, dim=0)
        x_eog = torch.squeeze(x_eog, dim=0)
        y_temp = torch.squeeze(y_temp, dim=0)

        epochs_to_keep = math.floor(x_eeg.shape[1]/200)*200
        x_eeg = x_eeg[:,0:epochs_to_keep,...]
        x_eog = x_eog[:,0:epochs_to_keep,...]
        y_temp = y_temp[0:epochs_to_keep]
        
        y_pred, loss = self.predict_single_channel(x_eeg, x_eog, y_temp)

        a = acc(torch.Tensor(y_pred), torch.Tensor(y_temp))
        k = kappa(torch.Tensor(y_pred), torch.Tensor(y_temp))
        f = f1(torch.Tensor(y_pred), torch.Tensor(y_temp), average=False)
        
        self.validation_step_loss.append(loss)
        self.validation_step_acc.append(a)
        self.validation_step_kap.append(k)
        self.validation_step_f1.append(f)

    def ensemble_testing(self, x_eegs, x_eogs):
        eegshape = x_eegs.shape
        eogshape = x_eogs.shape

        num_eegs = eegshape[0]
        num_eogs = eogshape[0]
        
        num_epochs = eegshape[1]
        
        assert eegshape[1] == eogshape[1]

        votes = torch.zeros(num_epochs, 5)
        
        # Get prediction for every possible eeg+eog combination
        for i in range(num_eegs):
            for p in range(num_eogs):
                x_eeg = x_eegs[i,...]
                x_eog = x_eogs[p,...]

                x_temp = torch.stack([x_eeg, x_eog], dim=0)
                
                assert x_eeg.shape == x_eog.shape
                assert x_temp.shape[0] == 2

                x_temp = x_temp.swapaxes(0,1)
                x_temp = torch.squeeze(x_temp)

                num_windows = num_epochs - 200 + 1
                
                # Predict on every window of the record and sum the probabilities
                for ii in range(num_windows):
                    window = x_temp[ii:ii+200, ...]

                    pred = self(torch.unsqueeze(window.float(), 0))
                    pred = torch.squeeze(pred)
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    pred = pred.cpu()

                    votes[ii:ii+200] = torch.add(votes[ii:ii+200], pred)

        votes = torch.argmax(votes, axis=1)

        if torch.cuda.is_available():
            votes = votes.cuda()

        return votes
    
    def test_step(self, batch, _):
        x_eegs, x_eogs, y_temp, tags = batch
        
        x_eegs = torch.squeeze(x_eegs, dim=0)
        x_eogs = torch.squeeze(x_eogs, dim=0)
        y_temp = torch.squeeze(y_temp, dim=0)
        
        epochs_to_keep = math.floor(x_eegs.shape[1]/200)*200

        x_eegs_cut = x_eegs[:,0:epochs_to_keep,...]
        x_eogs_cut = x_eogs[:,0:epochs_to_keep,...]
        y_temp_cut = y_temp[0:epochs_to_keep]
        
        y_pred_ensemble = self.ensemble_testing(x_eegs, x_eogs)
        y_pred_single, _ = self.predict_single_channel(x_eegs_cut, x_eogs_cut, y_temp_cut)
        
        tag = tags[0]
        tags = tag.split("/")

        log_test_step("results", self.logger.version, tags[0], tags[1], tags[2], channel_pred=y_pred_ensemble, single_pred=y_pred_single, labels=y_temp)


