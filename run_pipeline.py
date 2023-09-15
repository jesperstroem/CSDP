import torch
from torch.utils.data import DataLoader
from shared.pipeline.pipeline_dataset import PipelineDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.neptune import NeptuneLogger

import neptune as neptune
import yaml
from yaml.loader import SafeLoader
from neptune.utils import stringify_unsupported
from lightning_models.factories.concrete_model_factories import LSeqSleepNet_Factory, USleep_Factory
from lightning_models.factories.concrete_pipeline_factories import LSeqSleepNet_Pipeline_Factory, USleep_Pipeline_Factory

def main():  
    with open('pipeline_args.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        model = data['model']
        model_parameters = data['model_parameters']
        training = data['training']
        neptune = data['neptune']
        datasets = data['datasets']
    
    torch.set_float32_matmul_precision('high')
    accelerator = "gpu" if training["use_gpu"] == True else "cpu"

    if model == "lseq":
        model_fac = LSeqSleepNet_Factory()
        pipeline_fac = LSeqSleepNet_Pipeline_Factory()
    elif model == "usleep":
        model_fac = USleep_Factory()
        pipeline_fac = USleep_Pipeline_Factory()
    else:
        print("No valid model specified")
        exit()

    if training["use_pretrained"] == True:
        net = model_fac.create_pretrained_net(model_parameters, training, training["pretrained_path"])
    else:
        net = model_fac.create_new_net(model_parameters, training)    
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="valKap",
        min_delta=0.00,
        patience=training["early_stop_patience"],
        verbose=True,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    train_pipes = pipeline_fac.create_training_pipeline(training, datasets)
    val_pipes = pipeline_fac.create_validation_pipeline(training, datasets)
    test_pipes = pipeline_fac.create_test_pipeline(training, datasets)
    
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
            name=model,
            source_files=["pipeline_args.yaml", "run_pipeline.py"],
        )

    except:
        print("Error: No valid neptune logging credentials configured.")
        exit()

    trainer = pl.Trainer(logger=logger,
                         max_epochs=training["max_epochs"],
                         callbacks= callbacks,
                         accelerator=accelerator,
                         devices=training["devices"],
                         num_nodes=training["num_nodes"],
                         strategy="ddp")

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
    
    testset = PipelineDataset(pipes=test_pipes,
                              batch_size=None,
                              iterations=100000,
                              global_rank=trainer.global_rank,
                              world_size=trainer.world_size)
    
    if training["test"]==False:
        trainloader = DataLoader(trainset,
                                batch_size=training["batch_size"],
                                shuffle=False,
                                num_workers=1)
        
        valloader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)

        trainer.fit(net, trainloader, valloader)
    else:
        testloader = DataLoader(testset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)
        
        with torch.no_grad():
            net.eval()
            _ = trainer.test(net, testloader)
        
if __name__ == '__main__':
    main()
    
