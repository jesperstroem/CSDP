from setuptools import setup

setup(
    name="commonsleepdatapipeline",
    version="1.0.0",
    description="Package for data serving neural networks in automatic sleep staging",
    url="https://gitlab.au.dk/tech_ear-eeg/common-sleep-data-pipeline",
    author="Jesper Strøm",
    author_email="js@ece.au.dk",
    packages=[
        "csdp_pipeline.preprocessing",
        "csdp_pipeline.factories",
        "csdp_pipeline.pipeline_elements",
        "csdp_training",
        "csdp_training.lightning_models.factories"
    ],
    install_requires=["numpy", "scipy", "torch"],
)
