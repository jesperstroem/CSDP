from abc import ABC, abstractmethod
import pytorch_lightning as pl
from ml_architectures.lseqsleepnet.lseqsleepnet import LSeqSleepNet
from ml_architectures.lseqsleepnet.long_sequence_model import LongSequenceModel
from ml_architectures.lseqsleepnet.epoch_encoder import MultipleEpochEncoder
from ml_architectures.lseqsleepnet.classifier import Classifier
from ml_architectures.usleep.usleep import USleep
import pytorch_lightning as pl
from training.lightning_models.usleep import USleep_Lightning

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
                 ):
        self.lr = lr
        self.batch_size = batch_size

    def create_new_net(self) -> pl.LightningModule:
        inner = USleep()

        net = USleep_Lightning(inner,
                               self.lr,
                               self.batch_size)
        
        return net

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        pass

# class LSeqSleepNet_Factory(Model_Factory):
#     def create_new_net(self) -> pl.LightningModule:
#         model_args = model_args["lseq"]

#         inner = self.create_inner(model_args)

#         net = LSeqSleepNet_Lightning(inner,
#                                      model_args["lr"],
#                                      model_args["weight_decay"],
#                                      lr_red_factor=train_args["lr_reduction"],
#                                      lr_patience=train_args["lr_patience"])
#         return net

#     def create_pretrained_net(self) -> pl.LightningModule:
#         model_args = model_args["lseq"]

#         inner = self.create_inner()
        
#         net = LSeqSleepNet_Lightning.load_from_checkpoint(pretrained_path,
#                                                           lseqsleep=inner,
#                                                           lr=model_args["lr"],
#                                                           wd=model_args["weight_decay"])
#         return net

#     def create_inner(self) -> LSeqSleepNet:
#         sequences = model_args["sequences"]
#         classes = model_args["classes"]

#         F = model_args["F"]
#         M = model_args["M"]
#         num_channels = model_args["num_channels"]
#         minF = model_args["minF"]
#         maxF = model_args["maxF"]

#         samplerate = model_args["samplerate"]
#         K = model_args["K"]
#         B = model_args["B"]
#         lstm_hidden_size = model_args["lstm_hidden_size"]
#         fc_hidden_size = model_args["fc_hidden_size"]
#         classes = model_args["classes"]
#         attention_size = model_args["attention_size"]

#         enc_conf = MultipleEpochEncoder.Config(F,M,minF=minF,maxF=maxF,samplerate=samplerate,
#                                                seq_len=sequences, lstm_hidden_size=lstm_hidden_size,
#                                                attention_size=attention_size, num_channels=num_channels)

#         lsm_conf = LongSequenceModel.Config(K, B, lstm_input_size=lstm_hidden_size*2,
#                                             lstm_hidden_size=lstm_hidden_size)

#         clf_conf = Classifier.Config(lstm_hidden_size*2, fc_hidden_size, classes)
        
#         inner = LSeqSleepNet(enc_conf, lsm_conf, clf_conf)
#         return inner