from setuptools import setup

setup(
    name="commonsleepdatapipeline",
    version="0.1.0",
    description="Package for data serving neural networks in automatic sleep staging",
    url="https://gitlab.au.dk/tech_ear-eeg/common-sleep-data-pipeline",
    author="Jesper Strøm",
    author_email="js@ece.au.dk",
    packages=[
        "common_sleep_data_pipeline.preprocessing",
    ],
    install_requires=["numpy", "scipy"],
)
