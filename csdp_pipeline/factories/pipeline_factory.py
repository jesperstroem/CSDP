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
    def __init__(self, hdf5_base_path, split_path, trainsets, valsets, testsets):
        self.split_path = split_path
        self.hdf5_base_path = hdf5_base_path
        self.trainsets = trainsets
        self.valsets = valsets
        self.testsets = testsets

    def create_training_pipeline(self):
        train_pipes = [
            Sampler(
                self.hdf5_base_path,
                self.trainsets,
                self.split_path,
                split_type="train",
                num_epochs=35,
                subject_percentage=1.0,
            )
        ]
        return train_pipes

    def create_validation_pipeline(self):
        val_pipes = [
            Determ_sampler(
                self.hdf5_base_path,
                self.valsets,
                self.split_path,
                split_type="val",
                num_epochs=35,
            )
        ]

        return val_pipes

    def create_test_pipeline(self):
        test_pipes = [
            Determ_sampler(
                self.hdf5_base_path,
                self.testsets,
                self.split_path,
                split_type="test",
                num_epochs=35,
            )
        ]

        return test_pipes


class LSeqSleepNet_Pipeline_Factory(IPipeline_Factory):
    def __init__(
        self, hdf5_base_path, split_path, trainsets, valsets, testsets, num_epochs
    ):
        self.split_path = split_path
        self.hdf5_base_path = hdf5_base_path
        self.trainsets = trainsets
        self.valsets = valsets
        self.testsets = testsets
        self.num_epochs = num_epochs

    def create_training_pipeline(self):
        return [
            Sampler(
                self.hdf5_base_path,
                self.trainsets,
                self.split_path,
                split_type="train",
                num_epochs=self.num_epochs,
                subject_percentage=1.0,
            ),
            Resampler(source_sample=128, target_sample=100),
            Spectrogram(),  # default parameters match SeqSleepNet and derivatives
        ]

    def __evaluation_pipeline(self, split_type, datasets):
        return [
            Determ_sampler(
                self.hdf5_base_path,
                datasets,
                self.split_path,
                split_type=split_type,
                num_epochs=self.num_epochs,
            ),
            Resampler(source_sample=128, target_sample=100),
            Spectrogram(),
        ]

    def create_validation_pipeline(self):
        return self.__evaluation_pipeline("val", self.valsets)

    def create_test_pipeline(self):
        return self.__evaluation_pipeline("test", self.testsets)
