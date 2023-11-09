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
        "csdp_training",
        "csdp_datastore"
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