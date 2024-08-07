from abc import ABC, abstractmethod
from ml_architectures.lseqsleepnet.lseqsleepnet import LSeqSleepNet
from ml_architectures.lseqsleepnet.long_sequence_model import LongSequenceModel
from ml_architectures.lseqsleepnet.classifier import Classifier
from ml_architectures.lseqsleepnet.utils import make_lseqsleepnet_config
from ml_architectures.usleep.usleep import USleep
import pytorch_lightning as pl
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_training.lightning_models.lseqsleepnet import LSeqSleepNet_Lightning
import torch

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
                 progression_factor,
                 lr_patience = 50,
                 lr_factor = 0.5,
                 lr_minimum = 1e-7,
                 include_eog = True,
                 loss_weights = None,
                 ):
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum
        self.batch_size = batch_size
        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor
        self.include_eog = include_eog
        self.loss_weights = loss_weights

    def create_new_net(self) -> pl.LightningModule:
        net = USleep_Lightning(self.lr,
                               self.batch_size,
                               self.initial_filters,
                               self.complexity_factor,
                               self.progression_factor,
                               self.lr_patience,
                               self.lr_factor,
                               self.lr_minimum,
                               self.loss_weights,
                               self.include_eog)
        
        return net

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        net =  USleep_Lightning.load_from_checkpoint(pretrained_path,
                                                     lr=self.lr,
                                                     batch_size = self.batch_size,
                                                     lr_patience = self.lr_patience,
                                                     lr_factor = self.lr_factor,
                                                     lr_minimum = self.lr_minimum,
                                                     loss_weights = self.loss_weights,
                                                     map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        return net


class LSeqSleepNet_Factory(IModel_Factory):
    def __init__(self,
                 lr,
                 batch_size,
                 lr_patience = 50,
                 lr_factor = 0.5,
                 lr_minimum = 1e-7):
        self.lr = lr
        self.batch_size = batch_size
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_minimum = lr_minimum

    def create_new_net(self) -> pl.LightningModule:
        inner = self.__create_inner()

        lightning = LSeqSleepNet_Lightning(inner, 
                                           self.lr, 
                                           self.batch_size,
                                           self.lr_patience,
                                           self.lr_factor,
                                           self.lr_minimum)

        return lightning

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        inner = self.__create_inner()

        lightning = LSeqSleepNet_Lightning.load_from_checkpoint(pretrained_path,
                                                                inner,
                                                                lr=self.lr,
                                                                batch_size = self.batch_size,
                                                                lr_patience = self.lr_patience,
                                                                lr_factor = self.lr_factor,
                                                                lr_minimum = self.lr_minimum,
                                                                map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        return lightning
                                                                
    def __create_inner(self) -> LSeqSleepNet:
        lseq_config = make_lseqsleepnet_config(num_channels=2)

        lseq = LSeqSleepNet(lseq_config)

        return lseq
