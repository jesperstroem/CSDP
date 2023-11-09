from setuptools import setup

setup(
    name="commonsleepdatapipeline",
    version="1.0.0",
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
        "csdp_datastore.eesm"
    ],
    install_requires=["numpy", 
                      "scipy", 
                      "torch", 
                      "h5py", 
                      "mne==1.4.2", 
                      "numpy", 
                      "pandas", 
                      "pyarrow", 
                      "scikit_learn", 
                      "scipy", 
                      "setuptools", 
                      "wfdb"],
)