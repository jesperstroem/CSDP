"""
An implementation of SeqSleepNet using shared components from ml_architectures.
"""
import torch.nn as nn
from ..common.epoch_encoder import EpochEncoder
from .short_sequence_model import ShortSequenceModel
from .classifier import Classifier


class SeqSleepNet(nn.Module):
    """
    Implementation of the SeqSleepNet architecture.
    """

    class Config:
        def __init__(self, encoder_config, ssm_config, classifier_config):
            self.encoder_config = encoder_config
            self.ssm_config = ssm_config
            self.classifier_config = classifier_config

    def __init__(self, config):
        super().__init__()
        self.epoch_encoder = EpochEncoder(config.encoder_config)
        self.sequence_model = ShortSequenceModel(config.ssm_config)
        self.classifier = Classifier(config.classifier_config)

    def forward(self, x):
        # x is (Batch, Epoch, Channels, Sequence, Feature)
        x = self.epoch_encoder(x)

        # x is (Batch, Epoch, Feature)
        x = self.sequence_model(x)

        # x is (Batch, Epoch, Feature)
        x = self.classifier(x)

        # x is (Batch, Epoch, Logits)
        return x
