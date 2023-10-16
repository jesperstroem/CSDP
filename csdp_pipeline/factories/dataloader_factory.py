from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from csdp_pipeline.pipeline_elements.pipeline_dataset import PipelineDataset
from csdp_pipeline.factories.pipeline_factory import USleep_Pipeline_Factory
from csdp_training.utility import create_split_file

class IDataloader_Factory(ABC):
    def __init__(self, data_split_path, hdf5_base_path):        
        if data_split_path == None:
            self.data_split_path = create_split_file(hdf5_basepath=hdf5_base_path)
        else:
            self.data_split_path = data_split_path

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
                 data_split_path,
                 gradient_steps,
                 batch_size,
                 hdf5_base_path,
                 trainsets,
                 valsets,
                 testsets):
        super.__init__(data_split_path, hdf5_base_path)

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
    
class LSeqSleepNet_Dataloader_Factory(IDataloader_Factory):
    def __init__():
        pass

    def create_training_loader(self, num_workers):
        return super().create_training_loader(num_workers)
    
    def create_validation_loader(self, num_workers):
        return super().create_validation_loader(num_workers)
    
    def create_testing_loader(self, num_workers):
        return super().create_testing_loader(num_workers)
