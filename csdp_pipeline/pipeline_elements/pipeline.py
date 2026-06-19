from abc import ABC, abstractmethod

import torch

from csdp_pipeline.pipeline_elements.models import ISample
from csdp_pipeline.pipeline_elements.samplers import ISampler


class IPipe(ABC):
    @abstractmethod
    def process(x: ISample) -> ISample:
        pass


class Pipeline:
    def __init__(self, pipes: list[IPipe]):
        self.pipes = pipes

    def preprocess(self, batch) -> ISample:
        for _, p in enumerate(self.pipes):
            batch = p.process(batch)

        return batch


class PipelineConfiguration:
    def __init__(self, train: [IPipe] = [], val: [IPipe] = [], test: [IPipe] = []):
        """_summary_

        Args:
            train (IPipe], optional): A list of IPipe. Each pipe needs to implement the function "process", which transforms a single sample. Defaults to [] which means the sample is unchanged.
            val (IPipe], optional): A list of IPipe. Each pipe needs to implement the function "process", which transforms a single sample. Defaults to [] which means the sample is unchanged.
            test (IPipe], optional): A list of IPipe. Each pipe needs to implement the function "process", which transforms a single sample. Defaults to [] which means the sample is unchanged.
        """

        self.train_pipes = train
        self.val_pipes = val
        self.test_pipes = test

    def get_pipe_by_stage(self, stage: str):
        assert stage == "train" or stage == "val" or stage == "test"

        if stage == "train":
            return self.train_pipes
        elif stage == "val":
            return self.val_pipes
        else:
            return self.test_pipes


class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, sampler: ISampler, pipes: [IPipe]):
        self.sampler = sampler
        self.iterations = sampler.num_samples
        self.pipes = pipes

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        sample = self.sampler.get_sample(idx)

        for pipe in self.pipes:
            sample = pipe.process(sample)

        d = dict()

        d["eeg"] = sample.eeg
        d["eog"] = sample.eog
        d["labels"] = sample.labels
        d["tag"] = {
            "dataset": sample.tag.dataset,
            "subject": sample.tag.subject,
            "record": sample.tag.record,
            "eeg": sample.tag.eeg,
            "eog": sample.tag.eog,
        }
        # "start_idx": sample.tag.start_idx,
        # "end_idx": sample.tag.end_idx}

        return d
