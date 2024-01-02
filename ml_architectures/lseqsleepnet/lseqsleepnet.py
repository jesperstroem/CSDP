# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:40:59 2023

@author: repse
"""

import torch.nn as nn
import torch.nn.functional as F
from ml_architectures.lseqsleepnet.long_sequence_model import LongSequenceModel
from ml_architectures.common.epoch_encoder import EpochEncoder
from ml_architectures.lseqsleepnet.classifier import Classifier


class LSeqSleepNet(nn.Module):
    class Config:
        def __init__(self, encoder_config, lsm_config, classifier_config):
            self.encoder_config = encoder_config
            self.lsm_config = lsm_config
            self.classifier_config = classifier_config

    def __init__(self, config):
        super().__init__()
        self.epoch_encoder = EpochEncoder(config.encoder_config)
        self.sequence_model = LongSequenceModel(config.lsm_config)
        self.classifier = Classifier(config.classifier_config)

    def forward(self, x):
        # x is (Batch, Epoch, Channels, Sequence, Feature)
        x = self.epoch_encoder(x)

        # x is (Batch, Epoch, Feature)
        x = self.sequence_model(x)

        # x is (Batch, Epoch, Feature)
        x = self.classifier(x)

        # x is (Batch, Epoch, Probabilities)
        return x
