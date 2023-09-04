import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule
from h5py import File
from sklearn import preprocessing
from argparse import ArgumentParser
from joblib import Memory
import random
import numpy as np
import usleep_pytorch.utils
import h5py
import os
import sys
from h5py import File

from shared.pipeline.pipe import Pipeline
from shared.pipeline.sampler import Sampler
from shared.pipeline.determ_sampler import Determ_sampler
from shared.pipeline.augmenters import Augmenter
from shared.pipeline.pipeline_dataset import PipelineDataset
   

class TdbDataModule(LightningDataModule):
    def __init__(self, batch_size, iterations, datasets_path, fit_datasets, test_datasets, split_file, shuffle=True, drop_last=True, **kwargs) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.iterations = iterations
        self.datasets_path = datasets_path
        self.fit_datasets = fit_datasets
        self.test_datasets = test_datasets
        self.split_file = split_file
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.kwargs = kwargs


    def set_rank_and_worldsize(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def setup(self, stage="fit"):        
        if stage == "fit":
            self.train_dataset = PipelineDataset(
                [
                    Sampler(self.datasets_path, self.fit_datasets, self.split_file, split_type="train", num_epochs=35),
                    Augmenter(
                        min_frac=self.kwargs.get("min_frac"), 
                        max_frac=self.kwargs.get("max_frac"), 
                        apply_prob=self.kwargs.get("apply_prob"), 
                        sigma=self.kwargs.get("sigma"), 
                        mean=self.kwargs.get("mean"))
                ], self.batch_size, self.iterations, self.rank, self.world_size
            )
            self.val_dataset = PipelineDataset(
                [
                    Determ_sampler(self.datasets_path, self.fit_datasets, self.split_file, split_type="val", num_epochs=35)
                ], self.batch_size, self.iterations, self.rank, self.world_size)
        elif stage == "test":
            self.test_dataset = PipelineDataset(
                [
                    Determ_sampler(self.datasets_path, self.test_datasets, self.split_file, split_type="test", num_epochs=35)
                ], self.batch_size, self.iterations, self.rank, self.world_size)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1 # TODO: 16
        )
    
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1, # A whole night
            shuffle=False,
            num_workers=1
        )
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
