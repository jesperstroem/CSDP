# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

from argparse import ArgumentParser
import pickle
from datetime import datetime
import json
import torch
import torch.distributed as dist
import torch.nn as nn
import usleep_pytorch.utils as utils
from pytorch_lightning import LightningModule, Trainer
from sklearn import metrics
from neptune.utils import stringify_unsupported
from shared.pipeline.pipe import Pipeline
from shared.pipeline.sampler import Sampler
from shared.pipeline.determ_sampler import Determ_sampler
from shared.pipeline.augmenters import Augmenter
from shared.pipeline.pipeline_dataset import PipelineDataset

from torchmetrics.classification import MulticlassCohenKappa

from shared.utility import kappa, acc, f1, create_confusionmatrix, plot_confusionmatrix, log_test_step


def get_model(args):
    model = USleepModel(**vars(args))

    return model


class ConvBNELU(nn.Module):
    def __init__(
        self, 
        in_channels=2, 
        out_channels=6, 
        kernel_size=9, 
        dilation=1,
        ceil_pad=False
        ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2
        self.ceil_pad = ceil_pad

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0), # https://iq.opengenus.org/output-size-of-convolution/
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True
            ),
            nn.ELU(),
            nn.BatchNorm1d(self.out_channels)
        )

        self.ceil_padding = nn.Sequential(
            nn.ConstantPad1d(padding=(0,1), value=0)
        )

        nn.init.xavier_uniform_(self.layers[1].weight) # Initializing weights for the conv1d layer
        nn.init.zeros_(self.layers[1].bias) # Initializing biases as zeros for the conv1d layer


    def forward(self, x):
        if (self.ceil_pad) and (x.shape[2] % 2 == 1): # Pad 1 if dimension is uneven
            x = self.ceil_padding(x)
        
        x = self.layers(x)
        
        # Added padding after since chaning decoder kernel from 9 to 2 introduced mismatch
        if (self.ceil_pad) and (x.shape[2] % 2 == 1): # Pad 1 if dimension is uneven
            x = self.ceil_padding(x)
            
        return x


class Bottom(nn.Module): 
    def __init__(
        self, 
        in_channels=214, 
        out_channels=306, 
        kernel_size=9, 
        dilation=1
        ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ELU(),
            nn.BatchNorm1d(self.out_channels)
        )
        nn.init.xavier_uniform_(self.layers[1].weight) # Initializing weights for the conv1d layer
        nn.init.zeros_(self.layers[1].bias) # Initializing biases as zeros for the conv1d layer

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self, 
        filters=[6, 9, 11, 15, 20, 28, 40, 55, 77, 108, 152, 214],
        in_channels=2, 
        maxpool_kernel=2, 
        kernel_size=9, 
        dilation=1
        ):

        super().__init__()

        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernel = maxpool_kernel
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.depth = len(self.filters)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                    ConvBNELU(
                    in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                    out_channels=self.filters[k],
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    ceil_pad = True
                )
            ) for k in range(self.depth) 
        ]) 

        self.maxpools = nn.ModuleList([
            nn.MaxPool1d(self.maxpool_kernel) for k in range(self.depth)
        ])

        self.bottom = nn.Sequential( # 
            ConvBNELU(
                in_channels=self.filters[-1],
                out_channels=302,
                kernel_size=self.kernel_size
            )
        )


    def forward(self, x):
        shortcuts = [] # Residual connections
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        encoded = self.bottom(x) 

        return encoded, shortcuts

class Decoder(nn.Module):
    def __init__(
        self, filters=[214, 152, 108, 77, 55, 40, 28, 20, 15, 11, 9, 6],
        upsample_kernel=2,
        in_channels=302, 
        out_channels=428, 
        kernel_size=2
        ):

        super().__init__()

        self.filters = filters
        self.upsample_kernel = upsample_kernel
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.depth = len(self.filters)

        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=self.upsample_kernel),
                ConvBNELU(
                    in_channels=self.in_channels if k == 0 else self.filters[k - 1] * 2,
                    out_channels=self.filters[k],
                    kernel_size=self.kernel_size,
                    ceil_pad = True
                )
            ) for k in range(self.depth)
        ])

        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNELU(
                in_channels=self.filters[k] * 2,
                out_channels=self.filters[k] * 2,
                kernel_size=self.kernel_size,
                ceil_pad = True
                )
            ) for k in range(self.depth)
        ])


    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]): # [::-1] data is taken in reverse order
            z = upsample(z)
            
            if z.shape[2] != shortcut.shape[2]:
                z = utils.CropToMatch(z, shortcut)
            
            z = torch.cat([shortcut, z], dim=1)
            
            z = block(z)

        return z


class Dense(nn.Module):
    def __init__(
        self, 
        in_channels=12, 
        num_classes=6, 
        kernel_size=1
        ):

        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.dense = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels, 
                out_channels=self.num_classes, 
                kernel_size=self.kernel_size, 
                bias=True
                ),
            nn.Tanh()
        )

        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)


    def forward(self, z):
        z = self.dense(z)

        return z


class SegmentClassifier(nn.Module):
    def __init__(self, num_classes=5, avgpool_kernel=3840, conv1d_kernel=1):
        super().__init__()
        self.num_classes = num_classes
        self.avgpool_kernel = avgpool_kernel
        self.conv1d_kernel = conv1d_kernel

        self.avgpool = nn.AvgPool1d(self.avgpool_kernel)

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=self.num_classes + 1, out_channels=self.num_classes, kernel_size=self.conv1d_kernel),
            nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=self.conv1d_kernel)
            #nn.Softmax(dim=1) # We dont want softmax before xentropy loss
        )

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)


    def forward(self, z):
        z = self.avgpool(z)
        z = self.layers(z)

        return z


class USleepModel(LightningModule):
    def __init__(
        self,
        lr,
        batch_size,
        ensemble_window_size,
        lr_scheduler_factor,
        lr_scheduler_patience,
        monitor_metric,
        monitor_mode
    ):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.ensemble_window_size = ensemble_window_size
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dense = Dense()
        self.classifier = SegmentClassifier()

        self.loss = nn.CrossEntropyLoss(ignore_index=5)


    def forward(self, x):
        x, shortcuts = self.encoder(x)
        x = self.decoder(x, shortcuts)
        x = self.dense(x)

        return x


    def classify_segments(self, x):
        # Run through encoder+decoder
        z = self(x)

        # Classify decoded samples
        y = self.classifier(z)

        return y


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

        single_pred = self.classify_segments(xbatch.float())
        
        return single_pred
    
    
    def channels_prediction(self, x_eegs, x_eogs):
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
                #print(f"---------------------------- new channel combination: {i},{p}")
                x_eeg = x_eegs[:,i,...]
                x_eog = x_eogs[:,p,...]

                assert x_eeg.shape == x_eog.shape
                
                x_temp = torch.stack([x_eeg, x_eog], dim=0)
                x_temp = torch.squeeze(x_temp)
                x_temp = torch.unsqueeze(x_temp, 0)

                assert x_temp.shape[1] == 2
                assert x_temp.shape[0 ]
                
                pred = self.classify_segments(x_temp.float())
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.squeeze(pred)
                pred = pred.swapaxes(0,1)
                pred = pred.cpu()
                
                votes = torch.add(votes, pred)
                
                
                #for ii in range(num_windows):
                #    start_index = (ii * step_size)
                #    end_index = (start_index + window_len)
                #    
                #    window = x_temp[:,:,start_index:end_index]
                #    
                #    pred = self.classify_segments(window.float())
                #    pred = torch.nn.functional.softmax(pred, dim=1)
                #    pred = torch.squeeze(pred)
                #    pred = pred.swapaxes(0,1)
                #    pred = pred.cpu()
                #    
                #    votes[ii:ii+self.ensemble_window_size] = torch.add(
                #        votes[ii:ii+self.ensemble_window_size],
                #        pred
                #    )
                               
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
                #print(f"---------------------------- new channel combination: {i},{p}")
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
                    
                    pred = self.classify_segments(window.float())
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
    

    def training_step(self, batch, batch_idx):
        x_eeg, x_eog, ybatch, tags = batch

        xbatch = torch.cat((x_eeg, x_eog), dim=1)

        pred = self.classify_segments(xbatch.float())
        
        step_loss, _, _, _ = self.compute_train_metrics(pred, ybatch)
        
        return step_loss # We need to return loss as lightning does loss.backward() under the hood


    def training_epoch_end(self, training_step_outputs):
        all_outputs = self.all_gather(training_step_outputs)

        loss = [x['loss'] for x in all_outputs] 

        loss = torch.cat(loss)

        mean_loss = torch.mean(loss)
        
        if self.trainer.is_global_zero:
            self.log('trainLoss', mean_loss, rank_zero_only=True)
            
            self.trainer.save_checkpoint(f"{self.logger.save_dir}/usleep/{self.logger.version}/checkpoints/latest.ckpt")


    def validation_step(self, batch, batch_idx):
        # Step per record
        x_eeg, x_eog, ybatch, tags = batch
        
        pred = self.single_prediction(x_eeg, x_eog)
        
        step_loss, step_acc, step_kap, step_f1 = self.compute_train_metrics(pred, ybatch)

        self.log('valLoss', step_loss, sync_dist=True)
        self.log('valAcc', step_acc, sync_dist=True)
        self.log('valKap', step_kap, sync_dist=True)
        self.log('val_f1_c0', step_f1[0], sync_dist=True)
        self.log('val_f1_c1', step_f1[1], sync_dist=True)
        self.log('val_f1_c2', step_f1[2], sync_dist=True)
        self.log('val_f1_c3', step_f1[3], sync_dist=True)
        self.log('val_f1_c4', step_f1[4], sync_dist=True)
        
                
    def test_step(self, batch, batch_idx):
        # Step per record
        x_eeg, x_eog, ybatch, tags = batch
        
        single_pred = self.single_prediction(x_eeg, x_eog)
       # s_step_loss, s_step_acc, s_step_kap, s_step_f1 = self.compute_train_metrics(single_pred, ybatch)
       # print(f"s_step_kap: {s_step_kap}")
       # ensemble_pred = self.ensemble_prediction(x_eeg, x_eog)
       # e_step_acc, e_step_kap, e_step_f1 = self.compute_test_metrics(ensemble_pred, ybatch)
       # print(f"e_step_kap: {e_step_kap}")
        channels_pred = self.channels_prediction(x_eeg, x_eog)
       # c_step_acc, c_step_kap, c_step_f1 = self.compute_test_metrics(channels_pred, ybatch)
       # print(f"c_step_kap: {c_step_kap}")
        
        tag = tags[0]
        tags = tag.split("/")
        
        #log_test_step(self.result_file_location, self.logger.version, tags[0], tags[1], tags[2], None, single_pred, ybatch)
        log_test_step(self.result_basepath, self.logger.version, tags[0], tags[1], tags[2], channel_pred=channels_pred, single_pred=single_pred, labels=ybatch)
        
    def run_tests(self,
                  trainer,
                  dataloader,
                  result_basepath,
                  model_id):
        self.model_id = model_id
        self.result_basepath = result_basepath
        
        with torch.no_grad():
            self.eval()
            results = trainer.test(self, dataloader)
        
        return results    
        
    def get_pipes(self, training, datasets):
        aug = training["augmentation"]

        if aug["use"] == True:
            print("Running with augmentation")
            train_pipes = [Sampler(datasets["base_path"],
                                   datasets["train"],
                                   training["datasplit_path"],
                                   split_type="train",
                                   num_epochs=35,
                                   subject_percentage = training["subject_percentage"]),
                           Augmenter(
                               min_frac=aug["min_frac"], 
                               max_frac=aug["max_frac"], 
                               apply_prob=aug["apply_prob"], 
                               sigma=aug["sigma"],
                               mean=aug["mean"]
                           )]
        else:
            train_pipes = [Sampler(datasets["base_path"],
                                   datasets["train"],
                                   training["datasplit_path"],
                                   split_type="train",
                                   num_epochs=35,
                                   subject_percentage = training["subject_percentage"])]

        val_pipes = [Determ_sampler(datasets["base_path"],
                                    datasets["val"],
                                    training["datasplit_path"],
                                    split_type="val",
                                    num_epochs=35,
                                    subject_percentage = training["subject_percentage"])]

        test_pipes = [Determ_sampler(datasets["base_path"],
                             datasets["test"],
                             training["datasplit_path"],
                             split_type="test",
                             num_epochs=35)]

        return train_pipes, val_pipes, test_pipes
    
    @staticmethod
    def get_new_net(model_args, train_args):
        if train_args["use_pretrained"] == True:
            model = USleepModel.load_from_checkpoint(train_args["pretrained_path"])
        else:
            lr = model_args["lr"]
            batch_size = train_args["batch_size"]
            window_size = model_args["epochs"]
            lr_reduction = train_args["lr_reduction"]
            lr_patience = train_args["lr_patience"]

            net = USleepModel(lr,
                              batch_size,
                              window_size,
                              lr_reduction,
                              lr_patience,
                              "valKap",
                              "max")

        return net
    
    @staticmethod
    def get_pretrained_net(pretrained_path):
        net = USleepModel.load_from_checkpoint(pretrained_path)
        return net