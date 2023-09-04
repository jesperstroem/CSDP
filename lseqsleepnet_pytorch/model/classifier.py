import torch.nn as nn
import torch
from .basic_layers import FC

class Classifier(nn.Module):
    class Config():
        def __init__(self, fc_input_size, fc_hidden_size, fc_output_size):
            self.input_size = fc_input_size
            self.hidden_size = fc_hidden_size
            self.output_size = fc_output_size
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.output_size = config.output_size
        
        self.layers = nn.Sequential(
            FC(config.input_size, config.hidden_size, dropout = 0.1, activation="relu"),
            FC(config.hidden_size, config.hidden_size, dropout = 0.1, activation="relu"),
            FC(config.hidden_size, config.output_size, dropout = 0.0, activation="none")
        )
    
    def forward(self, x):
        # X should be (Batch, Epoch, Features)
        
        num_batches, num_epochs, num_features = x.shape
        
        # Flatten to (Epoch, Features)
        x = torch.reshape(x, (-1, num_features))
        
        x = self.layers(x)
        
        # Unflatten back to (Batch, Epoch, Features)
        x = torch.reshape(x, (-1, num_epochs, self.output_size))
  
        return x