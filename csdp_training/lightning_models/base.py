# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class Base_Lightning(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        batch_size,
        lr_patience,
        lr_factor,
        lr_minimum
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum
        self.training_step_outputs = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_kap = []
        self.validation_step_f1 = []
        self.loss = nn.CrossEntropyLoss(ignore_index=5)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

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
        
        self.validation_step_loss.clear()
        self.validation_step_acc.clear()
        self.validation_step_kap.clear()
        self.validation_step_f1.clear()