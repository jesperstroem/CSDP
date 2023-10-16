# CSDP - Common Sleep Data pipeline

This repository contains a pipeline for training neural networks to perform automatic sleep staging.

It is two part and consists of: data preprocessing/standardization and data serving.

The preprocessing can be used to transform most available PSG datasets to a standardized HDF5 file.

The data serving can be used for two neural networks with different training setups.

## Preparation

First of all, Git needs to be installed on your machine: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
Then you need a distribution of Anaconda, check here: https://www.anaconda.com/download/ 

Then run the following commands in shell:

git clone https://gitlab.au.dk/tech_ear-eeg/common-sleep-data-pipeline.git
cd common-sleep-data-pipeline
conda env create --file env.yml
conda activate csdp

The conda environment file assumes you run your code on a CUDA compatible system.

## How to preprocess and standardize your data

A lot of public PSG datasets are available for preprocessing.

The code does not download the data, so this is assumed to be performed in advance by users.

Configuration is specified in conf.yaml - the below example transforms ABC.
Use sample rate 100 for L-SeqSleepNet and 128 for U-Sleep.

```yaml
parameters:
 scale_and_clip: True
 output_sample_rate: 128
target_path:
 "/my/target/output/path/"
datasets:
 - name: ABC
   path: "/path/to/abc/data/basepath/"
```

Datasets is a list, so multiple datasets can be transformed in the same execution.
Datasets-"path" is the path to the downloaded raw data.

Read the following sections for download sources and exactly which paths to specify when transforming.

When the configuration is done, run the python script "transform.py" to transform the data to HDF5.

## Install repo as a package
If you are looking to use the pipeline for your own personal project with your own training loop, you can install this repo as a package.

Run the following command to install:

pip install git+https://gitlab.au.dk/tech_ear-eeg/common-sleep-data-pipeline.git

## Run demonstration of the pipeline
If you are looking to just see the pipeline in action and training your own model, follow these steps:

1. Edit the "usleep_args.yaml" or "lseq_args.yaml" file to the desired configuration.

2. Run "python run_usleep.py" or "python run_lseq.py" to run the configuration from the yaml file. This can either be training a new network from scratch, finetuning an existing one or testing an existing one. Test results are saved to a results folder.