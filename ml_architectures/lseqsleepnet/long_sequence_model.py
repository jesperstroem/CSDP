import torch.nn as nn
import torch
from ml_architectures.common.bn_blstm import BLSTMLayer
from ml_architectures.common.basic_layers import FC


class LongSequenceModel(nn.Module):
    class Config:
        def __init__(self, K, B, lstm_input_size, lstm_hidden_size, dropout_rate):
            self.K = K
            self.B = B
            self.lstm_input_size = lstm_input_size
            self.lstm_hidden_size = lstm_hidden_size
            self.dropout_rate = dropout_rate

    def __init__(self, config: Config):
        super().__init__()
        self.folder = SequenceFolder(config.K, config.B)
        self.unfolder = SequenceUnfolder()
        self.intra = SubsequenceModel(
            B=config.B,
            K=config.K,
            input_size=config.lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            dropout_rate=config.dropout_rate,
        )
        # Input K and B reverse because the matrix is transposed
        self.inter = SubsequenceModel(
            B=config.K,
            K=config.B,
            input_size=config.lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            dropout_rate=config.dropout_rate,
        )

    def forward(self, x):
        # x is (Batch, Epoch, Feature)

        x = self.folder(x)
        # (Batch, B, K, Feature)
        x = self.intra(x)

        # Transpose tensor to do inter modeling
        x = torch.transpose(x, 1, 2)

        # (Batch, B, K, Feature)
        x = self.inter(x)

        # Transpose again to get the original shape
        x = torch.transpose(x, 1, 2)

        # (Batch, B, K, Feature)
        x = self.unfolder(x)
        # (Batch, Epoch, Feature)
        return x


class SequenceFolder(nn.Module):
    def __init__(self, K, B):
        super().__init__()
        self.K = K  # folded height
        self.B = B  # folded width

    def forward(self, x):
        # Assumes (Batch, Epoch, Features)
        _, num_epochs, _ = x.shape
        if num_epochs != self.K * self.B:
            raise ValueError(f"Expected {self.K*self.B} epochs, got {num_epochs}")

        x = x.unflatten(1, sizes=(self.B, self.K))

        # Returns (Batch, B, K, Features)
        return x


class SequenceUnfolder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        return x


class SubsequenceModel(nn.Module):
    def __init__(self, K, B, input_size, hidden_size, dropout_rate):
        super().__init__()

        self.blstm = BLSTMLayer(input_size, hidden_size, dropout_rate)
        self.fc = FC(
            hidden_size * 2, input_size, dropout=dropout_rate, activation="none"
        )
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.B = B
        self.K = K

    def forward(self, x):
        # Assumes (Batch, B, K, Features)

        _, B, K, num_features = x.shape

        assert self.K == K
        assert self.B == B
        assert self.input_size == num_features

        # Flatten to (Batch*B, K, Features)
        x = torch.reshape(x, (-1, K, num_features))

        # OBS. using "raw" input for residual connection matches Huy's code
        # but does not match the description in the paper
        residual = x

        # BLSTM
        x = self.blstm(x)

        # FC layer
        # First flatten to (Batch*B*K, Features)
        x = torch.reshape(x, (-1, self.hidden_size * 2))
        x = self.fc(x)

        # Layer Normalization
        # TODO: Is this correct with the dimension?
        x = torch.layer_norm(x, [self.hidden_size * 2])
        # torch.layer_norm()

        # Unflatten to (Batch*B, K, Features)
        x = torch.reshape(x, (-1, K, self.hidden_size * 2))

        # Residual connection
        # TODO: Is this the correct way to do residual?
        x += residual

        # Unflatten to (Batch, B, K, Features)
        x = torch.reshape(x, (-1, B, K, self.hidden_size * 2))

        return x
