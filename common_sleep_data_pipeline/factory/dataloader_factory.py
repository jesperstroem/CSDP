from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from common_sleep_data_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset
from common_sleep_data_pipeline.factory.pipeline_factory import USleep_Pipeline_Factory

class IDataloader_Factory(ABC):
    @abstractmethod
    def create_training_loader(self, num_workers):
        pass

    @abstractmethod
    def create_validation_loader(self, num_workers):
        pass
    
    @abstractmethod
    def create_testing_loader(self, num_workers):
        pass

class USleep_Dataloader_Factory(IDataloader_Factory):
    def __init__(self,
                 gradient_steps,
                 batch_size,
                 hdf5_base_path,
                 trainsets,
                 valsets,
                 testsets,
                 data_split_path = None):
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size

        self.fac = USleep_Pipeline_Factory(hdf5_base_path,
                                            data_split_path,
                                            trainsets,
                                            valsets,
                                            testsets)

    def create_training_loader(self, num_workers = 1):
        pipes = self.fac.create_training_pipeline()
        dataset = PipelineDataset(pipes, self.gradient_steps*self.batch_size)
        trainloader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
        return trainloader
    
    def create_validation_loader(self, num_workers = 1):
        pipes = self.fac.create_validation_pipeline()
        dataset = PipelineDataset(pipes, len(pipes[0].records))
        valloader = DataLoader(dataset,
                               batch_size = 1,
                               shuffle = False,
                               num_workers = num_workers)
        return valloader

    def create_testing_loader(self, num_workers = 1):
        pipes = self.fac.create_test_pipeline()
        dataset = PipelineDataset(pipes=pipes,
                                  iterations=len(pipes[0].records))
        
        testloader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers)
        return testloader