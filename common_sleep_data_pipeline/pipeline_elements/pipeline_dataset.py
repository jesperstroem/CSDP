
import torch
from .pipe import Pipeline

'''
Iterable dataset to read batches of data from chunks of spectrogram files.
'''

class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, pipes, iterations):
        self.pipeline = Pipeline(pipes)
        self.iterations = iterations
        
    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        return self.pipeline.get_batch(idx)