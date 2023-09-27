# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from shared.utility import kappa, acc, f1, log_test_step
import time

def get_model(args):
    model = USleep_Lightning(**vars(args))

    return model

class USleep_Lightning(LightningModule):
    def __init__(
        self,
        usleep,
        lr,
        batch_size,
        ensemble_window_size,
        lr_scheduler_factor,
        lr_scheduler_patience,
        monitor_metric,
        monitor_mode,
    ):
        super().__init__()
        self.usleep = usleep
        self.lr = lr
        self.batch_size = batch_size
        self.ensemble_window_size = ensemble_window_size
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.training_step_outputs = []
        self.validation_step_loss = []
        self.validation_step_acc = []
        self.validation_step_kap = []
        self.validation_step_f1 = []

        self.save_hyperparameters(ignore=['usleep'])

        self.loss = nn.CrossEntropyLoss(ignore_index=5)

    def forward(self, x):
        return self.usleep(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=self.monitor_mode,
                                                             factor=self.lr_scheduler_factor,
                                                             patience=self.lr_scheduler_patience,
                                                             verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.monitor_metric
        }
    
    
    def compute_train_metrics(self, y_pred, y_true):
        y_pred = torch.swapdims(y_pred, 1, 2)
        y_pred = torch.reshape(y_pred, (-1, 5))
        y_true = torch.flatten(y_true)

        loss = self.loss(y_pred, y_true)

        y_pred = torch.argmax(y_pred, dim=1)
        
        accu = acc(y_pred, y_true)
        kap = kappa(y_pred, y_true, 5)
        f1_score = f1(y_pred, y_true, average=False)
        
        return loss, accu, kap, f1_score
    
    
    def compute_test_metrics(self, y_pred, y_true):
        y_true = torch.flatten(y_true)
        
        accu = acc(y_pred, y_true)
        kap = kappa(y_pred, y_true, 5)
        f1_score = f1(y_pred, y_true, average=False)
        
        return accu, kap, f1_score
    
            
    def single_prediction(self, x_eeg, x_eog):
        chan1 = x_eeg[:,0,...]
        chan2 = x_eog[:,0,...]
        
        xbatch = torch.stack([chan1, chan2], dim=1)

        single_pred = self(xbatch.float())
        
        return single_pred
    
    
    def channels_prediction(self, x_eegs, x_eogs):
        eegshape = x_eegs.shape
        eogshape = x_eogs.shape
        
        num_eegs = eegshape[1]
        num_eogs = eogshape[1]
        
        assert eegshape[2] == eogshape[2]
        
        signal_len = eegshape[2]
        num_epochs = int(signal_len / 128 / 30)
        
        votes = torch.zeros(num_epochs, 5) # fordi vi summerer løbende
        
        for i in range(num_eegs):
            for p in range(num_eogs):
 
                x_eeg = x_eegs[:,i,...]
                x_eog = x_eogs[:,p,...]

                assert x_eeg.shape == x_eog.shape
                
                x_temp = torch.stack([x_eeg, x_eog], dim=0)
                x_temp = torch.squeeze(x_temp)
                x_temp = torch.unsqueeze(x_temp, 0)

                assert x_temp.shape[1] == 2
                assert x_temp.shape[0 ]
                
                pred = self(x_temp.float())
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.squeeze(pred)
                pred = pred.swapaxes(0,1)
                pred = pred.cpu()
                
                votes = torch.add(votes, pred)
 
        votes = torch.argmax(votes, axis=1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        votes = votes.to(device)
        
        return votes
    
            
    def ensemble_prediction(self, x_eegs, x_eogs):
        window_len = self.ensemble_window_size * 128 * 30
        epoch_step_size = 1 # Should always be one
        step_size = epoch_step_size * 128 * 30 # Number  of epochs per step * sample rate * seconds
        
        eegshape = x_eegs.shape
        eogshape = x_eogs.shape
        
        num_eegs = eegshape[1]
        num_eogs = eogshape[1]
        
        assert eegshape[2] == eogshape[2]
        
        signal_len = eegshape[2]
        num_epochs = int(signal_len / 128 / 30)
        num_windows = int(signal_len/step_size) - int(window_len/step_size) + 1
        
        votes = torch.zeros(num_epochs, 5) # fordi vi summerer løbende
        
        for i in range(num_eegs):
            for p in range(num_eogs):
                x_eeg = x_eegs[:,i,...]
                x_eog = x_eogs[:,p,...]

                assert x_eeg.shape == x_eog.shape
                
                x_temp = torch.stack([x_eeg, x_eog], dim=0)
                x_temp = torch.squeeze(x_temp)
                x_temp = torch.unsqueeze(x_temp, 0)

                assert x_temp.shape[1] == 2
                assert x_temp.shape[0 ]
                
                for ii in range(num_windows):
                    start_index = (ii * step_size)
                    end_index = (start_index + window_len)
                    
                    window = x_temp[:,:,start_index:end_index]
                    
                    pred = self(window.float())
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    pred = torch.squeeze(pred)
                    pred = pred.swapaxes(0,1)
                    pred = pred.cpu()
                    
                    votes[ii:ii+self.ensemble_window_size] = torch.add(
                        votes[ii:ii+self.ensemble_window_size],
                        pred
                    )
                               
        votes = torch.argmax(votes, axis=1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        votes = votes.to(device)
        
        return votes
    

    def training_step(self, batch, idx):
        x_eeg, x_eog, ybatch, _ = batch

        xbatch = torch.cat((x_eeg, x_eog), dim=1)
        print(f"Batch shape: {xbatch.shape}, batch index: {idx}")
        xbatch = xbatch.float()
        
        #print("First: %s seconds ---" % (time.time() - start_time))
        pred = self.forward(xbatch)
        #print("Second: %s seconds ---" % (time.time() - start_time))

        step_loss, _, _, _ = self.compute_train_metrics(pred, ybatch)
        #print("Third: %s seconds ---" % (time.time() - start_time))

        self.training_step_outputs.append(step_loss)

        #print(f"Step loss: {step_loss} from {self.global_rank}")

        return step_loss # We need to return loss as lightning does loss.backward() under the hood


    def on_train_epoch_end(self):
        all_outputs = self.training_step_outputs
        
        mean_loss = torch.mean(torch.stack(all_outputs, dim=0))
        
        self.log('trainLoss', mean_loss, batch_size=self.batch_size, rank_zero_only=True)    
        
        self.trainer.save_checkpoint(f"{self.logger.save_dir}/usleep/{self.logger.version}/checkpoints/latest.ckpt")

        self.training_step_outputs.clear()


    def validation_step(self, batch, _):
        # Step per record
        x_eeg, x_eog, ybatch, _ = batch
        
        pred = self.single_prediction(x_eeg, x_eog)
        
        step_loss, step_acc, step_kap, step_f1 = self.compute_train_metrics(pred, ybatch)

        self.validation_step_loss.append(step_loss)
        self.validation_step_acc.append(step_acc)
        self.validation_step_kap.append(step_kap)
        self.validation_step_f1.append(step_f1)

        #print(f"Step loss: {step_loss} from {self.global_rank}")
        #print(pred.shape)
        #print(step_loss)
        #sync_dist = True
        #batch_size = 1

        #self.log('valLoss', step_loss, batch_size=batch_size, sync_dist=sync_dist)
        #self.log('valAcc', step_acc, batch_size=batch_size, sync_dist=sync_dist)
        #self.log('valKap', step_kap, batch_size=batch_size, sync_dist=sync_dist)
        #self.log('val_f1_c0', step_f1[0], batch_size=batch_size, sync_dist=sync_dist)
        #self.log('val_f1_c1', step_f1[1], batch_size=batch_size, sync_dist=sync_dist)
        #self.log('val_f1_c2', step_f1[2], batch_size=batch_size, sync_dist=sync_dist)
        #self.log('val_f1_c3', step_f1[3], batch_size=batch_size, sync_dist=sync_dist)
        #self.log('val_f1_c4', step_f1[4], batch_size=batch_size, sync_dist=sync_dist)

    def on_validation_epoch_end(self):

        all_losses = self.validation_step_loss
        all_acc = self.validation_step_acc
        all_kap = self.validation_step_kap    
        all_f1 = self.validation_step_f1

        #print(all_f1)
        #print(torch.stack(all_f1, dim=0))
        #print(torch.stack(all_f1, dim=1))
        mean_loss = torch.mean(torch.stack(all_losses, dim=0))
        mean_acc = torch.mean(torch.stack(all_acc, dim=0))
        mean_kap = torch.mean(torch.stack(all_kap, dim=0))
        
        mean_f1c0 = torch.mean(torch.stack(all_f1, dim=1)[0])
        mean_f1c1 = torch.mean(torch.stack(all_f1, dim=1)[1])
        mean_f1c2 = torch.mean(torch.stack(all_f1, dim=1)[2])
        mean_f1c3 = torch.mean(torch.stack(all_f1, dim=1)[3])
        mean_f1c4 = torch.mean(torch.stack(all_f1, dim=1)[4])
        
        batch_size=1
        sync_dist=True

        print(mean_acc)
        print(mean_kap)
        
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

                
    def test_step(self, batch, _):
        # Step per record
        x_eeg, x_eog, ybatch, tags = batch
        
        single_pred = self.single_prediction(x_eeg, x_eog)
        channels_pred = self.channels_prediction(x_eeg, x_eog)
        
        tag = tags[0]
        tags = tag.split("/")

        log_test_step("results", self.logger.version, tags[0], tags[1], tags[2], channel_pred=channels_pred, single_pred=single_pred, labels=ybatch)
