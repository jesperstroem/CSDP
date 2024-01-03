from setuptools import setup

setup(
    name="commonsleepdatapipeline",
    version="1.0.1",
    description="Package for data serving neural networks in automatic sleep staging",
    url="https://gitlab.au.dk/tech_ear-eeg/common-sleep-data-pipeline",
    author="Jesper Strøm",
    author_email="js@ece.au.dk",
    packages=[
        "csdp_pipeline",
        "csdp_pipeline.factories",
        "csdp_pipeline.pipeline_elements",
        "csdp_pipeline.preprocessing",
        "csdp_training",
        "csdp_training.lightning_models",
        "csdp_training.lightning_models.factories",
        "csdp_datastore",
        "csdp_datastore.dod",
        "csdp_datastore.isruc",
        "csdp_datastore.mass",
        "csdp_datastore.sdo",
        "csdp_datastore.sedf",
        "csdp_datastore.eesm",
        "ml_architectures",
        "ml_architectures.common",
        "ml_architectures.lseqsleepnet",
        "ml_architectures.usleep"
    ],
    install_requires=["numpy~=1.26.2", 
                      "scipy~=1.11.4", 
                      "torch~=2.0.1", 
                      "h5py~=3.10.0", 
                      "mne==1.4.2", 
                      "lightning~=2.1.3", 
                      "pandas~=2.1.4", 
                      "pyarrow~=14.0.2", 
                      "scikit_learn~=1.3.2", 
                      "scipy~=1.11.4", 
                      "setuptools~=68.2.2", 
                      "wfdb~=4.1.2"],
)