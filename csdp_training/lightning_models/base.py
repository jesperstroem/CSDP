# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from csdp_training.utility import kappa, acc, f1, plot_confusionmatrix, filter_unknowns
from sklearn.metrics import confusion_matrix
import numpy as np
from neptune.utils import stringify_unsupported

class Base_Lightning(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        batch_size,
        lr_patience,
        lr_factor,
        lr_minimum,
        loss_weights
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum
        self.loss_weights = loss_weights
        self.training_step_outputs = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_kap = []
        self.validation_step_f1 = []
        self.validation_preds = []
        self.validation_labels = []

        weights = torch.tensor(loss_weights) if loss_weights != None else None

        self.loss = nn.CrossEntropyLoss(weight=weights,
                                        ignore_index=5)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x.float()) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='max',
                                                             factor=self.lr_factor,
                                                             patience=self.lr_patience,
                                                             threshold=1e-4,
                                                             threshold_mode='rel',
                                                             cooldown=0,
                                                             min_lr=self.lr_minimum,
                                                             eps=1e-8,
                                                             verbose=True)

        return {
            'optimizer': optimizer,
            'monitor': 'valKap',
            'lr_scheduler': scheduler
        }
    
    def compute_train_metrics(self, y_pred, y_true):
        y_pred = torch.swapdims(y_pred, 1, 2)
        y_pred = torch.reshape(y_pred, (-1, 5))
        y_true = torch.flatten(y_true)

        loss = self.loss(y_pred, y_true)

        y_pred = torch.argmax(y_pred, dim=1)
        
        try:
            accu = acc(y_pred, y_true)
            kap = kappa(y_pred, y_true, 5)
            f1_score = f1(y_pred, y_true, average=False)
        except:
            accu = None
            kap = None
            f1_score = None
        
        return loss, accu, kap, f1_score
    
    
    def compute_test_metrics(self, y_pred, y_true):
        y_true = torch.flatten(y_true)
        
        accu = acc(y_pred, y_true)
        kap = kappa(y_pred, y_true, 5)
        f1_score = f1(y_pred, y_true, average=False)
        
        return accu, kap, f1_score

    def on_train_epoch_end(self):
        all_outputs = self.training_step_outputs
        
        mean_loss = torch.mean(torch.stack(all_outputs, dim=0))
        
        self.log('trainLoss', mean_loss, batch_size=self.batch_size, rank_zero_only=True)    

        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        all_losses = self.validation_step_loss
        all_acc = self.validation_step_acc
        all_kap = self.validation_step_kap    
        all_f1 = self.validation_step_f1
        all_preds = self.validation_preds
        all_labels = self.validation_labels

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        all_preds, all_labels = filter_unknowns(all_preds, all_labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        cm = confusion_matrix(all_labels, all_preds)

        mean_loss = torch.mean(torch.stack(all_losses, dim=0))
        mean_acc = torch.mean(torch.stack(all_acc, dim=0))
        mean_kap = torch.mean(torch.stack(all_kap, dim=0))
        
        mean_f1c0 = torch.mean(torch.stack(all_f1, dim=1)[0])
        mean_f1c1 = torch.mean(torch.stack(all_f1, dim=1)[1])
        mean_f1c2 = torch.mean(torch.stack(all_f1, dim=1)[2])
        mean_f1c3 = torch.mean(torch.stack(all_f1, dim=1)[3])
        mean_f1c4 = torch.mean(torch.stack(all_f1, dim=1)[4])
        
        batch_size=1
        
        self.log('valLoss', mean_loss, batch_size=batch_size, rank_zero_only=True)
        self.log('valAcc', mean_acc, batch_size=batch_size, rank_zero_only=True)
        self.log('valKap', mean_kap, batch_size=batch_size, rank_zero_only=True)
        self.log('val_f1_c0', mean_f1c0, batch_size=batch_size, rank_zero_only=True)
        self.log('val_f1_c1', mean_f1c1, batch_size=batch_size, rank_zero_only=True)
        self.log('val_f1_c2', mean_f1c2, batch_size=batch_size, rank_zero_only=True)
        self.log('val_f1_c3', mean_f1c3, batch_size=batch_size, rank_zero_only=True)
        self.log('val_f1_c4', mean_f1c4, batch_size=batch_size, rank_zero_only=True)

        cm = plot_confusionmatrix(cm, "")
        self.logger.experiment["training/val_cm"].append(stringify_unsupported(cm))

        self.validation_step_loss.clear()
        self.validation_step_acc.clear()
        self.validation_step_kap.clear()
        self.validation_step_f1.clear()
        self.validation_labels.clear()
        self.validation_preds.clear()
