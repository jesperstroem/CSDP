from common_sleep_data_pipeline.lightning_models.factories.model_factory import Model_Factory
from common_sleep_data_pipeline.lightning_models.lseqsleepnet import LSeqSleepNet_Lightning
from common_sleep_data_pipeline.lightning_models.usleep import USleep_Lightning
from ml_architectures.lseqsleepnet.lseqsleepnet import LSeqSleepNet
from ml_architectures.lseqsleepnet.long_sequence_model import LongSequenceModel
from ml_architectures.lseqsleepnet.epoch_encoder import MultipleEpochEncoder
from ml_architectures.lseqsleepnet.classifier import Classifier
from ml_architectures.usleep.usleep import USleep
import pytorch_lightning as pl

class USleep_Factory(Model_Factory):
    def create_new_net(self, model_args, train_args) -> pl.LightningModule:
        model_args = model_args["usleep"]

        lr = model_args["lr"]
        batch_size = train_args["batch_size"]
        progression_factor = model_args["progression_factor"]
        complexity_factor = model_args["complexity_factor"]

        inner = USleep(num_channels=2,
                       complexity_factor=complexity_factor,
                       progression_factor=progression_factor)

        net = USleep_Lightning(inner,
                               lr=lr,
                               batch_size=64)
        
        return net

    def create_pretrained_net(self, pretrained_path) -> pl.LightningModule:
        inner = USleep()
        lr = 0.0000001
        return USleep_Lightning.load_from_checkpoint(pretrained_path,
                                                     usleep=inner,
                                                     lr=lr,
                                                     batch_size=64)

class LSeqSleepNet_Factory(Model_Factory):
    def create_new_net(self, model_args, train_args) -> pl.LightningModule:
        model_args = model_args["lseq"]

        inner = self.create_inner(model_args)

        net = LSeqSleepNet_Lightning(inner,
                                     model_args["lr"],
                                     model_args["weight_decay"],
                                     lr_red_factor=train_args["lr_reduction"],
                                     lr_patience=train_args["lr_patience"])
        return net

    def create_pretrained_net(self, model_args, train_args, pretrained_path) -> pl.LightningModule:
        model_args = model_args["lseq"]

        inner = self.create_inner(model_args,train_args)
        
        net = LSeqSleepNet_Lightning.load_from_checkpoint(pretrained_path,
                                                          lseqsleep=inner,
                                                          lr=model_args["lr"],
                                                          wd=model_args["weight_decay"])
        return net

    def create_inner(self, model_args) -> LSeqSleepNet:
        sequences = model_args["sequences"]
        classes = model_args["classes"]

        F = model_args["F"]
        M = model_args["M"]
        num_channels = model_args["num_channels"]
        minF = model_args["minF"]
        maxF = model_args["maxF"]

        samplerate = model_args["samplerate"]
        K = model_args["K"]
        B = model_args["B"]
        lstm_hidden_size = model_args["lstm_hidden_size"]
        fc_hidden_size = model_args["fc_hidden_size"]
        classes = model_args["classes"]
        attention_size = model_args["attention_size"]

        enc_conf = MultipleEpochEncoder.Config(F,M,minF=minF,maxF=maxF,samplerate=samplerate,
                                               seq_len=sequences, lstm_hidden_size=lstm_hidden_size,
                                               attention_size=attention_size, num_channels=num_channels)

        lsm_conf = LongSequenceModel.Config(K, B, lstm_input_size=lstm_hidden_size*2,
                                            lstm_hidden_size=lstm_hidden_size)

        clf_conf = Classifier.Config(lstm_hidden_size*2, fc_hidden_size, classes)
        
        inner = LSeqSleepNet(enc_conf, lsm_conf, clf_conf)
        return inner
