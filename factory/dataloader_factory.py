from abc import ABC, abstractmethod
from factory.pipeline_factory import Pipeline_Factory
from torch.utils.data import DataLoader
from common_sleep_data_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset
from factory.pipeline_factory import USleep_Pipeline_Factory
from pathlib import Path
import yaml
from yaml.loader import SafeLoader

class IDataloader_Factory(ABC):
    @abstractmethod
    def create_training_loader(self):
        pass

    @abstractmethod
    def create_validation_loader(self):
        pass
    
    @abstractmethod
    def create_testing_loader(self):
        pass

class Dataloader_Factory(IDataloader_Factory):
    def __init__(self,
                 model,
                 gradient_steps,
                 batch_size,
                 num_workers,
                 data_split_path,
                 hdf5_base_path,
                 trainsets,
                 valsets,
                 testsets):
        self.model = model
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.num_workers = num_workers

        if model == "usleep":
            self.fac = USleep_Pipeline_Factory(hdf5_base_path,
                                               data_split_path,
                                               trainsets,
                                               valsets,
                                               testsets)

    def create_training_loader(self):
        pipes = self.fac.create_training_pipeline()
        dataset = PipelineDataset(pipes, self.gradient_steps*self.batch_size)
        trainloader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        return trainloader
    
    def create_validation_loader(self):
        pipes = self.fac.create_validation_pipeline()
        dataset = PipelineDataset(pipes, len(pipes[0].records))
        valloader = DataLoader(dataset,
                               batch_size = 1,
                               shuffle = False,
                               num_workers = self.num_workers)
        return valloader

    def create_testing_loader(self):
        pipes = self.fac.create_test_pipeline()
        dataset = PipelineDataset(pipes=pipes,
                                  iterations=len(pipes[0].records))
        
        testloader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        return testloader