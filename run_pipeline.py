import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers.neptune import NeptuneLogger
from common_sleep_data_pipeline.factory.dataloader_factory import USleep_Dataloader_Factory
from training.lightning_models.lightning_model_factory import USleep_Factory

import neptune as neptune

def main():

    torch.set_float32_matmul_precision('high')

    fac = USleep_Dataloader_Factory(gradient_steps=5,
                                    batch_size=64,
                                    num_workers=8,
                                    data_split_path="C:/Users/au588953/Git Repos/CSDP/common_sleep_data_pipeline/splits/usleep_split.json",
                                    hdf5_base_path="C:/Users/au588953/hdf5",
                                    trainsets=["dcsm"],
                                    valsets=["dcsm"],
                                    testsets=["dcsm"])
    
    mfac = USleep_Factory(lr = 0.0000001,
                          batch_size = 64,
                          num_channels = 2)
    
    net = mfac.create_new_net()
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
    # try:
    #     logger = NeptuneLogger(
    #         api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YzViZjJlYy00NDNhLTRhN2EtOGZmYy00NDEzODBmNTgxYzMifQ==",
    #         project="NTLAB/bigsleep",
    #         name="usleep",
    #         source_files=["pipeline_args.yaml", "run_pipeline.py"],
    #     )
    # except:
    #     print("Error: No valid neptune logging credentials configured.")
    #     exit()

    logger = True

    trainer = pl.Trainer(logger=logger,
                         max_epochs=100,
                         callbacks= callbacks,
                         accelerator="cpu",
                         devices=1,
                         num_nodes=1)

    trainer.fit(net, train_loader, val_loader)

        
if __name__ == '__main__':
    main()
    
