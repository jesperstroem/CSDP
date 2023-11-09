from ml_architectures.common.epoch_encoder import EpochEncoder
from .short_sequence_model import ShortSequenceModel
from .classifier import Classifier
from .seqsleepnet import SeqSleepNet

_SEQSLEEPNET_KEYWORDS = {
    "num_filters",
    "time_dim",
    "frequency_dim",
    "attention_size",
    "num_channels",
    "encoder_hidden_size",
    "sequence_hidden_size",
    "dropout_rate",
}


def make_seqsleepnet_config(**kwargs) -> SeqSleepNet.Config:
    """
    Get a default config choice. You can customize specific elements.

    Args:
        num_filters (int): The number of filters in the bank. Default is 32.
        time_dim (int): The number of time bins in the spectrograms. Default is 29.
        frequency_dim (int): The number of frequency bins in the spectrograms. Default is 129.
        attention_size (int): The size of the attention vector. Default is 64.
        num_channels (int): The number of input channels. Default is 1.
        dropout_rate (float): The rate of dropout on LSTM layer outputs.
        encoder_hidden_size: The hidden size of the encoder LSTM.
        sequence_hidden_size: The hidden size of the sequence LSTM.

    Returns:
        SeqSleepNet.Config : A SeqSleepNet config object.

    """
    if any(not keyword in _SEQSLEEPNET_KEYWORDS for keyword in kwargs):
        error_set = set(kwargs.keys()).difference(_SEQSLEEPNET_KEYWORDS)
        raise ValueError(f"Received invalid keyword(s) {error_set}.")

    encoder_kwargs = {
        "num_filters": 32,
        "time_dim": 29,
        "frequency_dim": 129,
        "attention_size": 64,
        "num_channels": 1,
        "dropout_rate": 0.2,
    }

    for keyword in kwargs:
        if keyword in encoder_kwargs:
            encoder_kwargs[keyword] = kwargs[keyword]

    if "encoder_hidden_size" in kwargs:
        encoder_kwargs["lstm_hidden_size"] = kwargs["encoder_hidden_size"]
    else:
        encoder_kwargs["lstm_hidden_size"] = 64

    ssm_kwargs = {
        "lstm_hidden_size": 64,
        "lstm_input_size": 2 * encoder_kwargs["lstm_hidden_size"],
        "dropout_rate": 0.2,
    }

    if "sequence_hidden_size" in kwargs:
        ssm_kwargs["lstm_hidden_size"] = kwargs["sequence_hidden_size"]

    if "dropout_rate" in kwargs:
        ssm_kwargs["dropout_rate"] = kwargs["dropout_rate"]

    classifier_kwargs = {
        "fc_input_size": 2 * ssm_kwargs["lstm_hidden_size"],
        "fc_output_size": 5,
    }

    return SeqSleepNet.Config(
        encoder_config=EpochEncoder.Config(**encoder_kwargs),
        ssm_config=ShortSequenceModel.Config(**ssm_kwargs),
        classifier_config=Classifier.Config(**classifier_kwargs),
    )
