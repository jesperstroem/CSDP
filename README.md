# CSDP - Common Sleep Data pipeline

This repository contains a pipeline for training neural networks to perform automatic sleep staging.

It is two part and consists of: data preprocessing/standardization and data serving.

The preprocessing can be used to transform most avaiable PSG datasets to a standardized HDF5 file.

The data serving can be used for two neural networks with customizable training setups.

## How to preprocess and standardize your data

A lot of public PSG datasets are available for preprocessing.

The code does not download the data, so this is assumed to be performed in advance by users.

Configuration is specified in conf.yaml - the below example transforms ABC:

```yaml
parameters:
 scale_and_clip: True
 output_sample_rate: 128
target_path:
 "/my/target/output/path"
datasets:
 - name: Abc
   path: "/path/to/abc/data/"
```

Datasets is a list, so multiple datasets can be transformed in the same execution.
Datasets-"path" is the path to the downloaded raw data.

Read the following sections for download sources and exactly which paths to specify when transforming.

When the configuration is done, run the python script "transform.py" to transform the data to HDF5.

### Dataset sources & documentation

#### Datasets from SleepData.org
A lot of datasets are available from https://sleepdata.org/, and so far 9 of them can be transformed with this repository.
They can be downloaded with the NSRR gem - look at their documentation and the script "download_scripts/download_sdo.sh" for inspiration on how to do so.

When downloaded, the path to specify is the one containing the "polysomnography" folder.

Datasets currently available for transformation:
- ABC
- CCSHS
- CFS
- CHAT
- HOMEPAP
- MESA
- MROS
- SHHS
- SOF

#### DOD-H & DOD-O
DOD-H: https://dreem-dod-h.s3.eu-west-3.amazonaws.com/index.html  
DOD-O: https://dreem-dod-o.s3.eu-west-3.amazonaws.com/index.html

For transformation, simply specify the local folder containing the files for each dataset.

#### SEDF-SC and SEDF-ST

Both datasets can be found here: https://physionet.org/content/sleep-edfx/1.0.0/

When transforming SEDF-SC, specify the local folder where the files from the serverside "sleep-cassette" folder resides.
When transforming SEDF-ST, specify the local folder where the files from the serverside "sleep-telemetry" folder resides.

#### ISRUC I, II and III

The data can be found here: https://sleeptight.isr.uc.pt/
<br></br>
To ensure the data is on the correct format, run the script "download_scripts/download_isruc.sh" in your desired data folder, to download and organize the data correctly.
Then for transformation, specify the path to one of the three folders indicating the subgroups: "subgroupI", "subgroupII" or "subgroupIII".

#### SVUH

The data can be found here: https://physionet.org/content/ucddb/1.0.0/
<br></br>
When transforming, specify a path to the folder that contains the .rec and .txt files.

#### PHYS

The data can be found here: https://physionet.org/content/challenge-2018/1.0.0/training/
<br></br>
For transformation, simply specify the local folder containing the files from the link.

#### DCSM

Data can be found here: https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html

For transformation, specify the folder, containing all the subfolders with records. Each subfolder should contain a "psg.h5" file and a "hypnogram.ids" file.

#### MASS

The data can be found here: http://ceams-carsm.ca/en/MASS/
<br></br>
Only SS1 and SS3 are available for transformation.
<br></br>
- TODO: Rewrite transformer logic, so it can deal with data from the source.

## How to use the pipeline

1. Edit the "args.yaml" file to the desired configuration.

2. Run "python train.py" to train a model with the configuration from the yaml file.

3. Run "python test.py" to test an existing model with the configuration from the yaml file. This creates result files in the results folder.

4. Run "python report.py --run_id--" to get a report of the results for the given run id. --run_id-- should match the appropriate foldername inside the results folder.