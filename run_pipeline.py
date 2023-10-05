import torch
from torch.utils.data import DataLoader
from common_sleep_data_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers.neptune import NeptuneLogger
from factory.dataloader_factory import Dataloader_Factory

import neptune as neptune

def main():

    torch.set_float32_matmul_precision('high')

    fac = Dataloader_Factory("usleep",
                             443,
                             64,
                             8,
                             "C:/Users/au588953/Git Repos/CSDP/shared/splits/usleep_split.json",
                             "C:/Users/au588953/hdf5",
                             ["dcsm"],
                             ["abc", "dcsm", "cfs"],
                             ["abc"])
    
    train_loader = fac.create_training_loader()
    val_loader = fac.create_validation_loader()

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="valKap",
        min_delta=0.00,
        patience=100,
        verbose=True,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    richbar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(monitor="valKap", mode="max")
    timer = Timer()

    callbacks = [early_stopping,
                 timer,
                 lr_monitor,
                 richbar,
                 checkpoint_callback]
    

    try:
        logger = NeptuneLogger(
            api_key=neptune["api_key"],
            project=neptune["project"],
            name="usleep",
            source_files=["pipeline_args.yaml", "run_pipeline.py"],
        )

    except:
        print("Error: No valid neptune logging credentials configured.")
        exit()

    trainer = pl.Trainer(logger=logger,
                         profiler=profiler,
                         max_epochs=training["max_epochs"],
                         callbacks= callbacks,
                         accelerator=accelerator,
                         devices=training["devices"],
                         num_nodes=training["num_nodes"])

    trainer.fit(net, train_loader, val_loader)

        
if __name__ == '__main__':
    main()
    
