from ml_architectures.common.epoch_encoder import EpochEncoder
from .long_sequence_model import LongSequenceModel
from .classifier import Classifier
from .lseqsleepnet import LSeqSleepNet

_LSEQSLEEPNET_KEYWORDS = {
    "num_filters",
    "time_dim",
    "frequency_dim",
    "attention_size",
    "num_channels",
    "encoder_hidden_size",
    "sequence_hidden_size",
    "dropout_rate",
    "folded_width",
    "folded_height",
}


def make_lseqsleepnet_config(**kwargs) -> LSeqSleepNet.Config:
    """
    Get a default config choice. You can customize specific elements.
    Note that the default sequene length is 200 epochs.

    Args:
        num_filters (int): The number of filters in the bank. Default is 32.
        time_dim (int): The number of time bins in the spectrograms. Default is 29.
        frequency_dim (int): The number of frequency bins in the spectrograms. Default is 129.
        attention_size (int): The size of the attention vector. Default is 64.
        num_channels (int): The number of input channels. Default is 1.
        dropout_rate (float): The rate of dropout on LSTM layer outputs.
        encoder_hidden_size (int): The hidden size of the encoder LSTM.
        sequence_hidden_size (int): The hidden size of the sequence LSTM.
        folded_height (int): The height of the folded sequence matrix (B). Default is 10.
        folded_width (int): The width of the folded sequence matrix (K). Default is 20.
        fc_hidden_size (int): The hidden dimension of classifier fc layers. Default is 512.
    Returns:
        LSeqSleepNet.Config : A LSeqSleepNet config object.

    """
    if any(not keyword in _LSEQSLEEPNET_KEYWORDS for keyword in kwargs):
        error_set = set(kwargs.keys()).difference(_LSEQSLEEPNET_KEYWORDS)
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

    lsm_kwargs = {
        "lstm_hidden_size": 64,
        "lstm_input_size": 2 * encoder_kwargs["lstm_hidden_size"],
        "dropout_rate": 0.2,
        "B": 10,
        "K": 20,
    }

    if "sequence_hidden_size" in kwargs:
        lsm_kwargs["lstm_hidden_size"] = kwargs["sequence_hidden_size"]

    if "dropout_rate" in kwargs:
        lsm_kwargs["dropout_rate"] = kwargs["dropout_rate"]

    if "folded_height" in kwargs:
        lsm_kwargs["B"] = kwargs["folded_height"]

    if "folded_width" in kwargs:
        lsm_kwargs["K"] = kwargs["folded_widtht"]

    classifier_kwargs = {
        "fc_input_size": 2 * lsm_kwargs["lstm_hidden_size"],
        "fc_output_size": 5,
        "fc_hidden_size": 512,
    }

    return LSeqSleepNet.Config(
        encoder_config=EpochEncoder.Config(**encoder_kwargs),
        lsm_config=LongSequenceModel.Config(**lsm_kwargs),
        classifier_config=Classifier.Config(**classifier_kwargs),
    )
