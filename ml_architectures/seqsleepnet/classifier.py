import torch.nn as nn
import torch


class Classifier(nn.Module):
    class Config:
        def __init__(self, fc_input_size, fc_output_size):
            self.input_size = fc_input_size
            self.output_size = fc_output_size

    def __init__(self, config: Config):
        super().__init__()

        self.output_size = config.output_size

        self.fc = nn.Linear(config.input_size, config.output_size)

    def forward(self, x):
        # X should be (Batch, Epoch, Features)

        _, num_epochs, num_features = x.shape

        # Flatten to (Epoch, Features)
        x = torch.reshape(x, (-1, num_features))

        x = self.fc(x)

        # Unflatten back to (Batch, Epoch, Features)
        x = torch.reshape(x, (-1, num_epochs, self.output_size))

        return x
