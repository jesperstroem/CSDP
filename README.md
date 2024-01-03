# CSDP - Common Sleep Data pipeline

This repository contains a pipeline for preprocessing and loading of PSG data to be used for the training of neural networks in automatic sleep staging.

The repository has three primary submodules:
- "csdp_datastore": Preprocessing of PSG datasets
- "csdp_pipeline": Dataloading of the preprocessed data
- "csdp_training": Pytorch Lightning modules of L-SeqSleepNet and U-Sleep to be used for automatic sleep scoring with the dataloaders.

## Install repo as a package
Run the following command to install:

```console
pip install git+https://github.com/jesperstroem/CSDP.git
```

## Downloading and preprocessing

Before you can use the dataloaders and lightning modules, you need to download and preprocess the raw data. So far, automatic download is only implemented for the datasets from https://sleepdata.org/. Note that to download the data from https://sleepdata.org/, you need a personal download token from their website, and you need the NSRR ruby gem installed: https://github.com/nsrr/nsrr-gem.

For the others, you need to download it yourself first and point to the location of the raw data. See below example.

```python

from csdp_datastore import ABC

raw_data_path = "/path/to/raw/data/location"
output_data_path = "/path/to/output/data/file"
download_token = "<token to download datasets from Sleepdata.org>"

a = ABC(dataset_path = raw_data_path,
        output_path = output_data_path,
        output_sample_rate = 128,
        download_token = download_token)

# This call can be omitted, if you already downloaded the data
a.download()

# This call starts the preprocessing
a.port_data()

```

## Use the dataloaders

To use the implemented pytorch dataloaders, look at the following example

"hdf5_base_path" is the root path to your preprocessed HDF5 files from the common datastore.
train/val/test-sets is a list of datasets in the root path, that you want to use in the dataloading. It should be the name of the file without the hdf5 extension.
"data_split_path" is a json file containing the configuration of train/validation/test subjects. Look in the csdp_pipeline/splits if you want examples. If you leave it as "None", all subjects will be used from the listed datasets. If you specify the parameter "create_random_split" as True, then a random split json file will be created and used.


```python

from csdp_pipeline.factories.dataloader_factory import USleep_Dataloader_Factory

dataloader_factory = USleep_Dataloader_Factory(gradient_steps=100,
                                               batch_size=64,
                                               hdf5_base_path="/root/path/to/datasets",
                                               trainsets=["abc", "cfs"],
                                               valsets=["abc", "cfs"],
                                               testsets=["chat"],
                                               data_split_path="/path/to/split/file"
                                               create_random_split=False)

train_loader = dataloader_factory.create_training_loader(num_workers=1)
val_loader = dataloader_factory.create_validation_loader(num_workers=1)
test_loader = dataloader_factory.create_testing_loader(num_workers=1)

```

## Use the lightning models

To also use the implemented pytorch lightning versions of U-Sleep, see the following example.

If you want a pretrained model, you need to specify a checkpoint. A checkpoint for u-sleep is available in the checkpoints folder.

```python

from csdp_training.lightning_models.factories.lightning_model_factory import USleep_Factory

model_factory = USleep_Factory(lr = 0.0001,
                               batch_size = 64,
                               initial_filters = 5,
                               complexity_factor = 1.67,
                               progression_factor = 2)

usleep = model_factory.create_new_net()
usleep_pretrained = model_factory.create_pretrained_net("/path/to/usleep/checkpoint/file")


```

## Making additions to the csdp_datastore submodule

To make an addition, you can either create a new class inheriting from the baseclass "BaseDataset" or you can inherit from a subclass. See the following examples of a "base" EESM class and a specialization inheriting from the EESM class. Check the baseclass documentation or the existing classes if in doubt on how to design the functions.

```python

class EESM(BaseDataset):
    def label_mapping(self):
        return {
            1: self.Labels.Wake,
            2: self.Labels.REM,
            3: self.Labels.N1,
            4: self.Labels.N2,
            5: self.Labels.N3,
            # Any other additions here
        }
        
    def dataset_name(self):
        return "eesm"

    def channel_mapping(self):
        return {
            "ELA": self.Mapping(self.EarEEGRef.ELA, self.EarEEGRef.REF),
            "ELB": self.Mapping(self.EarEEGRef.ELB, self.EarEEGRef.REF),
            # More channels can go here
        }    

    def list_records(self, basepath):

        paths_dict = dict()

        # Return value must be a dictionary
        # Every key is a subject ID, and the values are on the form (data_path, label_path), which must be absolute paths to the files containing data and labels.

        return paths_dict

    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()
        y = []

        # Return value must be on the form: (Dictionary, Labels)
        # The dictionary has a key for every channel, and the values are (channel_data, sampling_rate)
        
        return x, y

class EESM_Raw(EESM):
    
    # Overriding the name of the dataset
    def dataset_name(self):
        return "eesm_raw"


    # Overriding the logic for how each record is read from disk, but keeping all other logic
    def read_psg(self, record):
        psg_path, hyp_path = record

        x = dict()
        y = []

        return x, y

```
