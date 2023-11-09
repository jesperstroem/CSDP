# CSDP - Common Sleep Data pipeline

This repository contains a pipeline for preprocessing and loading of PSG data to be used for the training of neural networks in automatic sleep staging.

The repository has three submodules:
- datastore: Preprocessing
- pipeline: Dataloading
- training: Pytorch Lightning modules of L-SeqSleepNet and U-Sleep to be used for automatic sleep scoring

## Install repo as a package
If you are looking to use the pipeline for your own personal project, you can install this repo as a package.

Run the following command to install:

pip install git+https://github.com/jesperstroem/CSDP.git

## Downloading and preprocessing

TODO

## Use the dataloaders

To use the implemented pytorch dataloaders, look at the following example

"hdf5_base_path" is the root path to your preprocessed HDF5 files from the common datastore.
train/val/test-sets is a list of datasets in the root path, that you want to use in the dataloading. It should be the name of the file without the hdf5 extension.
"data_split_path" is a json file containing the configuration of train/validation/test subjects. Look in the csdp_pipeline/splits if you want examples. If you leave it as "None", all subjects will be used from the listed datasets. If you specify the parameter "create_random_split" as True, then a random split json file will be created and used.


```python

from csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory

dataloader_factory = USleep_Dataloader_Factory(gradient_steps=100,
                                               batch_size=64,
                                               hdf5_base_path="C:/Users/au588953/hdf5/",
                                               trainsets=["abc"],
                                               valsets=["abc"],
                                               testsets=["abc"],
                                               data_split_path="C:/Users/au588953/Git Repos/usleep-eareeg/splits/usleep_split.json",
                                               create_random_split=False)

train_loader = dataloader_factory.create_training_loader(num_workers=1)
val_loader = dataloader_factory.create_validation_loader(num_workers=1)
test_loader = dataloader_factory.create_testing_loader(num_workers=1)

```

## Use the lightning models

To also use the implemented pytorch lightning versions of U-Sleep, see the following example.

If you want a pretrained model, you need to specify a checkpoint. A checkpoint for u-sleep is available in the checkpoints folder.

Note: To use these you need lightning installed, and the model package from https://gitlab.au.dk/tech_ear-eeg/ml_architectures

```python

from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory

model_factory = USleep_Factory(lr = 0.0001,
                               batch_size = 64,
                               initial_filters = 5,
                               complexity_factor = 1.67,
                               progression_factor = 2)

usleep = model_factory.create_new_net()
usleep_pretrained = model_factory.create_pretrained_net("C:/Users/au588953/Git Repos/CSDP/checkpoints/best_usleep.ckpt")


```