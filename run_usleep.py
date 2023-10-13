import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers.neptune import NeptuneLogger
from common_sleep_data_pipeline.factory.dataloader_factory import USleep_Dataloader_Factory
from training.lightning_models.lightning_model_factory import USleep_Factory
from pathlib import Path
import neptune as neptune
import yaml
from yaml.loader import SafeLoader

file_path = Path(__file__).parent.absolute()
args_path = f"{file_path}/usleep_args.yaml"

with open(args_path) as f:
    data = yaml.load(f, Loader=SafeLoader)
    neptune_info = data["neptune"]

environment = data["environment"]
train_sets = data["train_sets"]
val_sets = data["val_sets"]
pretrained = data["pretrained"]
pretrained_path = data["pretrained_path"]

gradient_steps = data["gradient_steps"]
batch_size = data["batch_size"]
num_workers = data["num_workers"]

lr = data["lr"]
max_epochs = data["max_epochs"]
early_stop_patience = data["early_stop_patience"]

logging_enabled = neptune_info["logging"]
neptune_api_key = neptune_info["api_key"]
neptune_project = neptune_info["project"]
neptune_name = neptune_info["name"]

if environment == "LOCAL":
    hdf5_data_path = "C:/Users/au588953/hdf5"
    hdf5_split_path = "C:/Users/au588953/Git Repos/CSDP/common_sleep_data_pipeline/splits/usleep_split.json"
    accelerator = "cpu"

elif environment == "LUMI":
    hdf5_data_path = "/users/strmjesp/mnt"
    hdf5_split_path = ""
    accelerator = "gpu"

elif environment == "PRIME":
    hdf5_data_path = "/com/ecent/NOBACKUP/HDF5/usleep_data_big"
    hdf5_split_path = "/home/js/repos/common-sleep-data-pipeline/common_sleep_data_pipeline/shared/splits/usleep_split.json"
    accelerator = "gpu"

def main():
    torch.set_float32_matmul_precision('high')

    fac = USleep_Dataloader_Factory(gradient_steps=gradient_steps,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    data_split_path=hdf5_split_path,
                                    hdf5_base_path=hdf5_data_path,
                                    trainsets=train_sets,
                                    valsets=val_sets,
                                    testsets=[])
    
    mfac = USleep_Factory(lr = lr,
                          batch_size = batch_size)
    
    if pretrained == False:
        net = mfac.create_new_net()
    else:
        net = mfac.create_pretrained_net(pretrained_path)

    train_loader = fac.create_training_loader()
    val_loader = fac.create_validation_loader()

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="valKap",
        min_delta=0.00,
        patience=early_stop_patience,
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
    
    if logging_enabled == True:
        try:
            logger = NeptuneLogger(
                api_key=neptune_api_key,
                project=neptune_project,
                name=neptune_name,
                source_files=["usleep_args.yaml", "run_usleep.py"],
            )
        except:
            print("Error: No valid neptune logging credentials configured.")
            exit()
    else:
        logger = True

    trainer = pl.Trainer(logger=logger,
                         max_epochs=max_epochs,
                         callbacks= callbacks,
                         accelerator=accelerator,
                         devices=1,
                         num_nodes=1)

    trainer.fit(net, train_loader, val_loader)

        
if __name__ == '__main__':
    main()
    
