# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from .long_sequence_model import LongSequenceModel
from .epoch_encoder import MultipleEpochEncoder
from .classifier import Classifier
from shared.utility import dump, acc, kappa, plot_confusionmatrix, create_confusionmatrix, majority_vote, f1, log_test_step
import numpy as np
import math
import sys
from neptune.utils import stringify_unsupported
import pickle
import pytorch_lightning as pl

from shared.pipeline.pipeline_dataset import PipelineDataset
from shared.pipeline.resampler import Resampler
from shared.pipeline.sampler import Sampler
from shared.pipeline.determ_sampler import Determ_sampler
from shared.pipeline.spectrogram import Spectrogram
from shared.pipeline.augmenters import Augmenter

class LSeqSleepNet_Lightning(pl.LightningModule):
    def __init__(self, 
                 lseqsleep,
                 lr,
                 wd,
                 lr_red_factor = 0.5,
                 lr_patience = 20,
                 num_epochs = 200,
                 classes = 5):
        super().__init__()
        self.lseqsleep = lseqsleep
        self.lr = lr
        self.wd = wd
        self.lr_red_factor = lr_red_factor
        self.lr_patience = lr_patience
        self.num_epochs = num_epochs
        self.num_classes = classes
        
    def forward(self, x):
        return self.lseqsleep(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     eps=1e-07,
                                     weight_decay=self.wd)
        
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='max',
                                                             factor=self.lr_red_factor,
                                                             patience=self.lr_patience,
                                                             threshold=1e-4,
                                                             threshold_mode='rel',
                                                             cooldown=0,
                                                             min_lr=0,
                                                             eps=1e-8,
                                                             verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valKap'
        }
    
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
    
    def training_step(self, batch, batch_idx):
        x_eegs, x_eogs, y_temp, _ = batch

        x_temp = torch.cat([x_eegs, x_eogs], dim=1)
  
        assert x_temp.shape[1] == 2
        
        x_temp, y_temp = self.prep_training_batch(x_temp, y_temp)
       
        y_pred = self(x_temp)
        y_pred = torch.reshape(y_pred, (-1, 5))
        loss = F.cross_entropy(y_pred, y_temp, ignore_index=5)   

        return loss
    
    def training_epoch_end(self, training_step_outputs):
        all_outputs = self.all_gather(training_step_outputs)

        loss = [x['loss'] for x in all_outputs] 

        loss = torch.cat(loss)

        mean_loss = torch.mean(loss)
        
        if self.trainer.is_global_zero:
            print("Hello from rank zero in train")
            self.log('trainLoss', mean_loss, rank_zero_only=True)
            self.trainer.save_checkpoint(f"{self.logger.save_dir}/lseq/{self.logger.version}/checkpoints/latest.ckpt")
       
    def predict_single_channel(self, x_eegs, x_eogs, y_temp):
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
        loss = F.cross_entropy(y_pred, y_temp, ignore_index=5) 
        
        y_pred = torch.argmax(y_pred, dim=1)
        
        return y_pred, loss
    
    def validation_step(self, batch, batch_idx):
        x_eegs, x_eogs, y_temp, _ = batch

        x_eegs = torch.squeeze(x_eegs, dim=0)
        x_eogs = torch.squeeze(x_eogs, dim=0)
        y_temp = torch.squeeze(y_temp, dim=0)

        epochs_to_keep = math.floor(x_eegs.shape[1]/200)*200
        x_eegs = x_eegs[:,0:epochs_to_keep,...]
        x_eogs = x_eogs[:,0:epochs_to_keep,...]
        y_temp = y_temp[0:epochs_to_keep]
        
        y_pred, loss = self.predict_single_channel(x_eegs, x_eogs, y_temp)
        
        a = acc(torch.Tensor(y_pred), torch.Tensor(y_temp))
        k = kappa(torch.Tensor(y_pred), torch.Tensor(y_temp))
        f = f1(torch.Tensor(y_pred), torch.Tensor(y_temp), average=False)
        
        self.log('valLoss', loss, sync_dist=True, batch_size=1)
        self.log('valAcc', a, sync_dist=True, batch_size=1)
        self.log('valKap', k, sync_dist=True, batch_size=1)
        self.log('f1_c1', f[0], sync_dist=True, batch_size=1)
        self.log('f1_c2', f[1], sync_dist=True, batch_size=1)
        self.log('f1_c3', f[2], sync_dist=True, batch_size=1)
        self.log('f1_c4', f[3], sync_dist=True, batch_size=1)
        self.log('f1_c5', f[4], sync_dist=True, batch_size=1)
        
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
        votes = votes.cuda()

        return votes
    
    def test_step(self, batch, batch_idx):
        x_eegs, x_eogs, y_temp, tags = batch
        
        x_eegs = torch.squeeze(x_eegs, dim=0)
        x_eogs = torch.squeeze(x_eogs, dim=0)
        y_temp = torch.squeeze(y_temp, dim=0)
        
        epochs_to_keep = math.floor(x_eegs.shape[1]/200)*200

        x_eegs_cut = x_eegs[:,0:epochs_to_keep,...]
        x_eogs_cut = x_eogs[:,0:epochs_to_keep,...]
        y_temp_cut = y_temp[0:epochs_to_keep]
        
        y_pred_ensemble = self.ensemble_testing(x_eegs, x_eogs)
        y_pred_single, loss = self.predict_single_channel(x_eegs_cut, x_eogs_cut, y_temp_cut)
        
        tag = tags[0]
        tags = tag.split("/")

        log_test_step(self.result_basepath, self.logger.version, tags[0], tags[1], tags[2], channel_pred=y_pred_ensemble, single_pred=y_pred_single, labels=y_temp)

    def run_tests(self, trainer, dataloader, result_basepath, model_id):
        self.model_id = model_id
        self.result_basepath = result_basepath
        
        with torch.no_grad():
            self.eval()
            results = trainer.test(self, dataloader)
        
        return results
    
    def get_pipes(self, train_args, dataset_args):
        aug = train_args["augmentation"]
        
        if aug["use"] == True:
            print("Running with augmentation")
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   train_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=200,
                                   subject_percentage = train_args["subject_percentage"]),
                           Augmenter(
                               min_frac=aug["min_frac"], 
                               max_frac=aug["max_frac"], 
                               apply_prob=aug["apply_prob"], 
                               sigma=aug["sigma"],
                               mean=aug["mean"]
                           ),
                           Resampler(128, 100),
                           Spectrogram()]
        else:
            train_pipes = [Sampler(dataset_args["base_path"],
                                   dataset_args["train"],
                                   train_args["datasplit_path"],
                                   split_type="train",
                                   num_epochs=200,
                                   subject_percentage = train_args["subject_percentage"]),
                           Resampler(128, 100),
                           Spectrogram()]

        val_pipes = [Determ_sampler(dataset_args["base_path"],
                                    dataset_args["val"],
                                    train_args["datasplit_path"],
                                    split_type="val",
                                    num_epochs=200,
                                    subject_percentage = train_args["subject_percentage"]),
                     Resampler(128, 100),
                     Spectrogram()]
        
        test_pipes = [Determ_sampler(dataset_args["base_path"],
                             dataset_args["test"],
                             train_args["datasplit_path"],
                             split_type="test",
                             num_epochs=200),
              Resampler(128, 100),
              Spectrogram()]
        
        return train_pipes, val_pipes, test_pipes
    
    @staticmethod
    def get_inner(model_args, train_args):
        features = model_args["features"]
        epochs = model_args["epochs"]
        sequences = model_args["sequences"]
        classes = model_args["classes"]
        lr=model_args["lr"]
        seed=model_args["seed"]
        weight_decay=model_args["weight_decay"]
        F = model_args["F"]
        M = model_args["M"]
        num_channels = model_args["num_channels"]
        minF = model_args["minF"]
        maxF = model_args["maxF"]
        source_samplerate = model_args["source_samplerate"]
        samplerate = model_args["samplerate"]
        K = model_args["K"]
        B = model_args["B"]
        lstm_hidden_size = model_args["lstm_hidden_size"]
        fc_hidden_size = model_args["fc_hidden_size"]
        classes = model_args["classes"]
        attention_size = model_args["attention_size"]

        earlyStoppingPatience = train_args["early_stop_patience"]

        enc_conf = MultipleEpochEncoder.Config(F,M,minF=minF,maxF=maxF,samplerate=samplerate,
                                               seq_len=sequences, lstm_hidden_size=lstm_hidden_size,
                                               attention_size=attention_size, num_channels=num_channels)

        lsm_conf = LongSequenceModel.Config(K, B, lstm_input_size=lstm_hidden_size*2,
                                            lstm_hidden_size=lstm_hidden_size)

        clf_conf = Classifier.Config(lstm_hidden_size*2, fc_hidden_size, classes)
        
        inner = LSeqSleepNet(enc_conf, lsm_conf, clf_conf)
        return inner
    
    @staticmethod
    def get_new_net(model_args, train_args):
        inner = LSeqSleepNet_Lightning.get_inner(model_args,train_args)

        net = LSeqSleepNet_Lightning(inner,
                                     model_args["lr"],
                                     model_args["weight_decay"],
                                     lr_red_factor=train_args["lr_reduction"],
                                     lr_patience=train_args["lr_patience"])
        
        return net
    
    @staticmethod
    def get_pretrained_net(model_args, train_args, pretrained_path):
        inner = LSeqSleepNet_Lightning.get_inner(model_args,train_args)
        
        net = LSeqSleepNet_Lightning.load_from_checkpoint(pretrained_path,
                                                          lseqsleep=inner,
                                                          lr=model_args["lr"],
                                                          wd=model_args["weight_decay"])
        return net
        
        
class LSeqSleepNet(nn.Module):
    class Config():
        def __init__(self, encoder_conf, lsm_conf, clf_conf):
            self.encoder_conf = encoder_conf
            self.lsm_conf = lsm_conf
            self.clf_conf = clf_conf

    def __init__(self, enc_conf, lsm_conf, clf_conf):
        super().__init__()
        self.epoch_encoder = MultipleEpochEncoder(enc_conf)
        self.sequence_model = LongSequenceModel(lsm_conf)
        self.classifier = Classifier(clf_conf)
        
    def forward(self, x):
        # x is (Batch, Epoch, Channels, Sequence, Feature)
        x = self.epoch_encoder(x)
        
        # x is (Batch, Epoch, Feature)        
        x = self.sequence_model(x)
        
        # x is (Batch, Epoch, Feature)
        x = self.classifier(x)
        
        # x is (Batch, Epoch, Probabilities)
        return x







