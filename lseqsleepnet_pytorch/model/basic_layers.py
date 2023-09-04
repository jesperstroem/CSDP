import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 activation = "relu"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = nn.Linear(input_size, output_size, device=self.device)
        self.activation = activation.lower()
        self.dropout = nn.Dropout(dropout)
        self.dropout_prob = dropout
    def forward(self, x):
        x = self.layer(x)
        
        if self.activation == "relu":
            func = nn.ReLU()
            x = func(x)
        elif self.activation == "softmax":
            func = nn.Softmax(dim=1)
            x = func(x)
        
        if(self.dropout_prob > 0.0):
            x = self.dropout(x)
        
        return x 