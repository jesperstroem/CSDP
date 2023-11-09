import torch.nn as nn
import torch
from ml_architectures.common.bn_blstm import BLSTMLayer
from ml_architectures.common.filterbank_utils import FilterbankUtilities


class EpochEncoder(nn.Module):
    class Config:
        def __init__(
            self,
            num_filters,
            frequency_dim,
            time_dim,
            lstm_hidden_size,
            attention_size,
            num_channels,
            dropout_rate,
        ):
            self.num_filters = num_filters
            self.frequency_dim = frequency_dim
            self.time_dim = time_dim
            self.lstm_hidden_size = lstm_hidden_size
            self.attention_size = attention_size
            self.num_channels = num_channels
            self.dropout_rate = dropout_rate

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.filterbank = LearnableFilterbank(
            num_filters=config.num_filters,
            num_channels=config.num_channels,
            frequency_dim=config.frequency_dim,
        )
        self.BLSTM = BLSTMLayer(
            input_size=config.num_filters * config.num_channels,
            hidden_size=config.lstm_hidden_size,
            dropout=config.dropout_rate,
        )
        # Bidirectional lstm returns output of size 2 * hidden size
        self.attention = AttentionLayer(
            feature_size=2 * config.lstm_hidden_size,
            time_dim=config.time_dim,
            attention_size=config.attention_size,
        )

    def forward(self, x):
        # Assumes (Batch, Epoch, Channels, Sequence, Feature)
        _, num_epochs, num_channels, time_dim, frequency_dim = x.shape

        if num_channels != self.num_channels:
            raise ValueError(
                f"Expected number of channels is {self.num_channels}. Got {num_channels}."
            )

        # Flatten to (Epoch, Channels, Sequence, Feature)
        x = torch.reshape(x, (-1, num_channels, time_dim, frequency_dim))

        # Process
        x = self.filterbank(x)
        x = self.BLSTM(x)
        x = self.attention(x)

        # Unflatten to (Batch, Epoch, Feature)
        x = torch.reshape(x, (-1, num_epochs, self.config.lstm_hidden_size * 2))
        return x


class LearnableFilterbank(nn.Module):
    def __init__(self, num_filters, num_channels, frequency_dim):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.frequency_dim = frequency_dim

        filter_shape = FilterbankUtilities.linear_tri_filterbank(
            num_filters=num_filters,
            frequency_dim=frequency_dim,
        )

        triangular_matrix = torch.tensor(filter_shape, dtype=torch.float)
        self.triangular_matrix = nn.Parameter(triangular_matrix, requires_grad=True)

        self.filter_weights = nn.Parameter(
            torch.randn(self.num_filters, self.num_channels)
        )

    def forward(self, x):
        num_epochs, num_channels, _, _ = x.shape

        chnls = []

        # Do filter for each channel
        for c in range(num_channels):
            data = x[:, c, :, :]

            filterbank = torch.multiply(
                torch.sigmoid(self.filter_weights[:, c]),
                self.triangular_matrix,
            )

            data = torch.matmul(data, filterbank)
            chnls.append(data)

        # Concatenate the outputs in the M dimension, so that final size is (num_epochs, 29, M*num_channels)
        x = torch.cat(chnls, dim=2)

        return x


class AttentionLayer(nn.Module):
    # From Kaare's implementation
    def __init__(self, feature_size, time_dim, attention_size):
        super().__init__()
        self.attweight_w = torch.nn.Parameter(torch.randn(feature_size, attention_size))
        self.attweight_b = torch.nn.Parameter(torch.randn(attention_size))
        self.attweight_u = torch.nn.Parameter(torch.randn(attention_size))
        self.feature_size = feature_size
        self.time_dim = time_dim

    def forward(self, x):
        v = torch.tanh(
            torch.matmul(torch.reshape(x, [-1, self.feature_size]), self.attweight_w)
            + torch.reshape(self.attweight_b, [1, -1])
        )
        vu = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, self.time_dim])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x = torch.sum(x * torch.reshape(alphas, [-1, self.time_dim, 1]), 1)
        return x
