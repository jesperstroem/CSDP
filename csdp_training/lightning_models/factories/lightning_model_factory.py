from abc import ABC, abstractmethod
import pytorch_lightning as pl
from ml_architectures.lseqsleepnet.lseqsleepnet import LSeqSleepNet
from ml_architectures.lseqsleepnet.long_sequence_model import LongSequenceModel
from ml_architectures.lseqsleepnet.epoch_encoder import MultipleEpochEncoder
from ml_architectures.lseqsleepnet.classifier import Classifier
from ml_architectures.usleep.usleep import USleep
import pytorch_lightning as pl
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_training.lightning_models.lseqsleepnet import LSeqSleepNet_Lightning

class IModel_Factory(ABC):
    @abstractmethod
    def create_new_net(self) -> pl.LightningModule:
        pass

    @abstractmethod
    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        pass

class USleep_Factory(IModel_Factory):
    def __init__(self,
                 lr,
                 batch_size,
                 initial_filters,
                 complexity_factor,
                 progression_factor
                 ):
        self.lr = lr
        self.batch_size = batch_size
        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor

    def create_new_net(self) -> pl.LightningModule:
        inner = USleep(num_channels=2,
                       initial_filters=self.initial_filters,
                       complexity_factor=self.complexity_factor,
                       progression_factor=self.progression_factor)

        net = USleep_Lightning(inner,
                               self.lr,
                               self.batch_size,
                               self.initial_filters,
                               self.complexity_factor,
                               self.progression_factor)
        
        return net

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        inner = USleep(num_channels=2,
                       initial_filters=self.initial_filters,
                       complexity_factor=self.complexity_factor,
                       progression_factor=self.progression_factor)

        net =  USleep_Lightning.load_from_checkpoint(pretrained_path,
                                                     usleep=inner,
                                                     lr=self.lr,
                                                     batch_size = self.batch_size)
        return net


class LSeqSleepNet_Factory(IModel_Factory):
    def create_new_net(self) -> pl.LightningModule:
        return None

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        return None

    def __create_inner(self) -> LSeqSleepNet:
        return None