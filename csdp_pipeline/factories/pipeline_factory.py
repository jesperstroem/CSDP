from abc import ABC, abstractmethod
from csdp_pipeline.pipeline_elements.sampler import Sampler
from csdp_pipeline.pipeline_elements.augmenters import Augmenter
from csdp_pipeline.pipeline_elements.resampler import Resampler
from csdp_pipeline.pipeline_elements.spectrogram import Spectrogram
from csdp_pipeline.pipeline_elements.determ_sampler import Determ_sampler

class IPipeline_Factory(ABC):
    @abstractmethod
    def create_training_pipeline(self):
        pass

    @abstractmethod
    def create_validation_pipeline(self):
        pass
    
    @abstractmethod
    def create_test_pipeline(self):
        pass

class USleep_Pipeline_Factory(IPipeline_Factory):
    def __init__(self,
                 hdf5_base_path,
                 split_path,
                 trainsets,
                 valsets,
                 testsets):
        self.split_path = split_path
        self.hdf5_base_path = hdf5_base_path
        self.trainsets = trainsets
        self.valsets = valsets
        self.testsets = testsets

    def create_training_pipeline(self):
        train_pipes = [Sampler(self.hdf5_base_path,
                                self.trainsets,
                                self.split_path,
                                split_type="train",
                                num_epochs=35,
                                subject_percentage = 1.0)]
        return train_pipes

    def create_validation_pipeline(self):
        val_pipes = [Determ_sampler(self.hdf5_base_path,
                            self.valsets,
                            self.split_path,
                            split_type="val",
                            num_epochs=35,
                            single_channels = True,
                            subject_percentage = 1.0)]
        
        return val_pipes

    def create_test_pipeline(self):
        test_pipes = [Determ_sampler(self.hdf5_base_path,
                                     self.testsets,
                                     self.split_path,
                                     split_type="test",
                                     num_epochs=35)]
        
        return test_pipes
    
class LSeqSleepNet_Pipeline_Factory(IPipeline_Factory):
    def __init__(self):
        pass

    def create_training_pipeline(self):
        return super().create_training_pipeline()
    
    def create_validation_pipeline(self):
        return super().create_validation_pipeline()
    
    def create_test_pipeline(self):
        return super().create_test_pipeline()