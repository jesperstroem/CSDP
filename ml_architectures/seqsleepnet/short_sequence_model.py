import torch.nn as nn


class ShortSequenceModel(nn.Module):
    class Config:
        def __init__(self, lstm_input_size, lstm_hidden_size, dropout_rate):
            self.lstm_input_size = lstm_input_size
            self.lstm_hidden_size = lstm_hidden_size
            self.dropout_rate = dropout_rate

    def __init__(self, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, x):
        # x is (Batch, Epoch, Feature)
        x, _ = self.lstm(x)  # obs. hidden states implicitly 0

        # x is (Batch, Feature)
        x = self.dropout(x)

        return x
