import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers.neptune import NeptuneLogger
from csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory, LSeqSleepNet_Dataloader_Factory
from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory, LSeqSleepNet_Factory
from pathlib import Path
import neptune as neptune
import yaml
from yaml.loader import SafeLoader
import os

file_path = Path(__file__).parent.absolute()
args_path = f"{file_path}/training_args.yaml"

with open(args_path) as f:
    data = yaml.load(f, Loader=SafeLoader)

model_type = data["model"]
neptune_info = data["neptune"]
train_sets = data["train_sets"]
val_sets = data["val_sets"]
test_sets = data["test_sets"]
pretrained = data["pretrained"]
pretrained_path = data["pretrained_path"]

test = True

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

hdf5_data_path = data["hdf5_base_path"]
hdf5_split_path = data["hdf5_split_path"]
accelerator = data["accelerator"]

def main():
    torch.set_float32_matmul_precision('high')

    if model_type == "usleep":
        parameters = data["usleep_parameters"]

        fac = USleep_Dataloader_Factory(gradient_steps=gradient_steps,
                                        batch_size=batch_size,
                                        hdf5_base_path=hdf5_data_path,
                                        trainsets=train_sets,
                                        valsets=val_sets,
                                        testsets=test_sets,
                                        data_split_path=hdf5_split_path)
        
        mfac = USleep_Factory(lr = lr,
                              batch_size = batch_size,
                              initial_filters=parameters["initial_filters"],
                              complexity_factor=parameters["complexity_factor"],
                              progression_factor=parameters["progression_factor"])
        
    elif model_type == "lseqsleepnet":
        fac = LSeqSleepNet_Dataloader_Factory()
        mfac = LSeqSleepNet_Factory()
    else:
        print("No valid model configuration")
        exit()
    
    if pretrained == False:
        net = mfac.create_new_net()
    else:
        net = mfac.create_pretrained_net(pretrained_path)

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
    
    # I hate this, but Lightning has no better way to change logging directory :(
    org = os.getcwd()

    if neptune_info["logging_folder"] != None:
        os.chdir(neptune_info["logging_folder"])

    if logging_enabled == True:
        try:
            existing_run_id = neptune_info["existing_run_id"]
            source_files = [f"{file_path}/training_args.yaml", f"{file_path}/run_training.py"]
            mode = "sync"

            if existing_run_id != None:
                existing_run = neptune.init_run(project=neptune_project,
                                                api_token=neptune_api_key,
                                                with_id=existing_run_id,
                                                name=neptune_name,
                                                source_files=source_files,
                                                mode=mode,)
                
                logger = NeptuneLogger(
                    run=existing_run
                )
            else:
                logger = NeptuneLogger(
                    api_key=neptune_api_key,
                    project=neptune_project,
                    name=neptune_name,
                    source_files=source_files,
                    mode = mode
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
    
    if test == True:
        with torch.no_grad():
            net.eval()

            test_loader = fac.create_testing_loader(num_workers=1)
            _ = trainer.test(net, test_loader)
    else:
        train_loader = fac.create_training_loader(num_workers=num_workers)
        val_loader = fac.create_validation_loader(num_workers=num_workers)

        trainer.fit(net, train_loader, val_loader)

    os.chdir(org)

        
if __name__ == '__main__':
    main()
    
