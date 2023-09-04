import os

from shared.utility import log_dataset_splits
from usleep_pytorch.usleep import USleepModel
from lseqsleepnet_pytorch.model.lseqsleepnet import LSeqSleepNet_Lightning
import torch
from torch.utils.data import DataLoader
from shared.pipeline.pipeline_dataset import PipelineDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import neptune.new as neptune
import json
import yaml
from yaml.loader import SafeLoader
import importlib
from neptune.utils import stringify_unsupported

def main():  
    with open('pipeline_args.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        model = data['model']
        lseq = data['lseq']
        usleep = data['usleep']
        training = data['training']
        neptune = data['neptune']
        datasets = data['datasets']
    
    torch.set_float32_matmul_precision('high')    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model == "lseq":
        if training["use_pretrained"] == True:
            net = LSeqSleepNet_Lightning.get_pretrained_net(lseq, training, training["pretrained_path"])
        else:    
            net = LSeqSleepNet_Lightning.get_new_net(lseq, training)
    elif model == "usleep":
        if training["use_pretrained"] == True:
            net = USleepModel.get_pretrained_net(training["pretrained_path"])
        else:
            net = USleepModel.get_new_net(usleep, training)
    else:
        print("No valid model specified")
        exit()
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="valKap",
        min_delta=0.00,
        patience=training["early_stop_patience"],
        verbose=True,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    base = datasets["base_path"]
    split_file = training["datasplit_path"]
    
    with open(split_file, "r") as split:
        splitdata = json.load(split)
    
    train_pipes, val_pipes, _ = net.get_pipes(training, datasets)
    
    iterations=training["iterations"]
    richbar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(monitor="valKap", mode="max")

    callbacks = [early_stopping,
                 lr_monitor,
                 richbar,
                 checkpoint_callback]
    
    try:
        logger = NeptuneLogger(
            api_key=neptune["api_key"],
            project=neptune["project"],
            name=model
        )

        logger.log_hyperparams(stringify_unsupported(lseq))
        logger.log_hyperparams(stringify_unsupported(usleep))
        logger.log_hyperparams(stringify_unsupported(training))
        logger.log_hyperparams(stringify_unsupported(neptune))
    except:
        print("Error: No valid neptune logging credentials configured.")
        exit()
    
    trainer = pl.Trainer(logger=logger,
                         max_epochs=training["max_epochs"],
                         callbacks= callbacks,
                         accelerator="gpu",
                         devices=training["devices"],
                         num_nodes=training["num_nodes"],
                         strategy="ddp_find_unused_parameters_false")

    trainset = PipelineDataset(pipes=train_pipes,
                               batch_size=None,
                               iterations=iterations,
                               global_rank=trainer.global_rank,
                               world_size=trainer.world_size)

    valset = PipelineDataset(pipes=val_pipes,
                             batch_size=None,
                             iterations=100000,
                             global_rank=trainer.global_rank,
                             world_size=trainer.world_size)

    trainloader = DataLoader(trainset,
                             batch_size=training["batch_size"],
                             shuffle=False,
                             num_workers=1)
    
    valloader = DataLoader(valset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=1)

    trainer.fit(net, trainloader, valloader)
        
if __name__ == '__main__':
    main()
    
