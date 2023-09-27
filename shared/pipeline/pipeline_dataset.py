
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import timeit

from .pipe import Pipeline

'''
Iterable dataset to read batches of data from chunks of spectrogram files.
'''

class PipelineDatasetV2(torch.utils.data.Dataset):
    def __init__(self, pipes, iterations):
        self.pipeline = Pipeline(pipes)
        self.iterations = iterations
        
    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        batch = self.pipeline.get_batch(idx)

        x_eegs, x_eogs, ybatch, tags = batch
        
        return (x_eegs, x_eogs, ybatch, tags)



class PipelineDataset(torch.utils.data.IterableDataset):
    def __init__(self, pipes, iterations, global_rank, world_size):
        super(PipelineDataset, self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterations = iterations
        self.pipes = pipes
        self.global_rank = global_rank
        self.world_size = world_size
       
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id 
        num_workers = worker_info.num_workers
        global_rank = self.global_rank
        world_size = self.world_size
      
        return iter(PipeIterator(self.pipes, self.iterations, worker_id, num_workers, global_rank, world_size))

class PipeIterator:
    def __init__(self,
                 pipes,
                 iterations,
                 worker_id,
                 num_workers,
                 global_rank,
                 world_size):
        self.iterations = iterations
        self.pipeline = Pipeline(pipes)
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.global_rank = global_rank
        self.world_size = world_size
        
    def __iter__(self):
        self.cnt = (self.global_rank*self.num_workers) + self.worker_id

        return self      
    
    def __next__(self):
        if self.cnt >= self.iterations:
            raise StopIteration      

        while True:
            batch = self.pipeline.get_batch(self.cnt)
               
            self.cnt += (self.num_workers * self.world_size)
               
            if batch == -2:
                raise StopIteration
            if batch == -1:
                continue
            else:
                break

        x_eegs, x_eogs, ybatch, tags = batch
        #print(f"Count: {self.cnt}, tag: {tags}")
        
        return (x_eegs, x_eogs, ybatch, tags)
