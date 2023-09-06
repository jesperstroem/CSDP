from abc import ABC, abstractmethod
import pytorch_lightning as pl

class Model_Factory(ABC):
    @abstractmethod
    def create_new_net(self, model_args, train_args) -> pl.LightningModule:
        pass

    @abstractmethod
    def create_pretrained_net(self, model_args, train_args, pretrained_path) -> pl.LightningModule:
        pass