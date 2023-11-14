# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

import torch
from csdp_training.lightning_models.lseqsleepnet import Base_Lightning
from csdp_training.utility import kappa, acc, f1, log_test_step

class USleep_Lightning(Base_Lightning):
    def __init__(
        self,
        usleep,
        lr,
        batch_size,
        initial_filters,
        complexity_factor,
        progression_factor
    ):
        super().__init__(usleep, lr, batch_size)

        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor
                
        self.save_hyperparameters(ignore=['usleep'])
    
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

    def training_step(self, batch, idx):
        x_eeg, x_eog, ybatch, _ = batch

        xbatch = torch.cat((x_eeg, x_eog), dim=1)

        xbatch = xbatch.float()

        pred = self.forward(xbatch)

        step_loss, _, _, _ = self.compute_train_metrics(pred, ybatch)

        self.training_step_outputs.append(step_loss)

        return step_loss

    def validation_step(self, batch, _):
        # Step per record
        x_eeg, x_eog, ybatch, _ = batch
        
        if any(dim == 0 for dim in x_eog.shape):
            print("Found no EOG channel, duplicating EEG instead")
            x_eog = x_eeg
        
        pred = self.single_prediction(x_eeg, x_eog)
        
        step_loss, step_acc, step_kap, step_f1 = self.compute_train_metrics(pred, ybatch)

        self.validation_step_loss.append(step_loss)
        self.validation_step_acc.append(step_acc)
        self.validation_step_kap.append(step_kap)
        self.validation_step_f1.append(step_f1)
                
    def test_step(self, batch, _):
        # Step per record
        x_eeg, x_eog, ybatch, tags = batch

        print(x_eeg.shape)
        print(x_eog.shape)
        
        if any(dim == 0 for dim in x_eog.shape):
            print("Found no EOG channel, duplicating EEG instead")
            x_eog = x_eeg

        ybatch = torch.flatten(ybatch)
        single_pred = self.single_prediction(x_eeg, x_eog)
        channels_pred = self.channels_prediction(x_eeg, x_eog)
        
        tag = tags[0]
        tags = tag.split("/")

        kap = kappa(channels_pred, ybatch, 5)
        print(tag)
        print(kap)
        
        log_test_step("results", self.logger.version, tags[0], tags[1], tags[2], single_pred=single_pred, channel_pred=channels_pred, labels=ybatch)
