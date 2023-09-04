import torch.nn as nn
import torch
from .bn_blstm import BLSTM_Layer, BLSTM_Layer_Torch
from shared.utility import dump
from .filterbank_shape import FilterbankShape

class MultipleEpochEncoder(nn.Module):
    class Config():
        def __init__(self, F, M, minF, maxF, samplerate,
                     seq_len, lstm_hidden_size,
                     attention_size, num_channels):
            self.F = F
            self.M = M
            self.minF = minF
            self.maxF = maxF
            self.samplerate = samplerate
            self.seq_len = seq_len
            self.lstm_hidden_size = lstm_hidden_size
            self.attention_size = attention_size
            self.num_channels = num_channels
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.encoder = SingleEpochEncoder(config.F, config.M, config.minF, config.maxF, config.samplerate,
                                          config.seq_len, config.lstm_hidden_size,
                                          config.attention_size, config.num_channels)
        
    def forward(self, x):
        # Assumes (Batch, Epoch, Channels, Sequence, Feature)
        num_batches, num_epochs, num_channels, num_sequences, num_features = x.shape
        
        assert num_channels == self.num_channels
        #dump(x)
        # Flatten to (Epoch, Sequence, Feature)
        
        x = torch.reshape(x, (-1, num_channels, num_sequences, num_features))
        #dump(x)
        x = self.encoder(x)
        
        #dump(x)
        # Unflatten to (Batch, Epoch, Feature)
        x = torch.reshape(x, (-1, num_epochs, self.config.lstm_hidden_size*2))
        #dump(x)
        return x

# Encodes one epoch
class SingleEpochEncoder(nn.Module):
    def __init__(self, F,M,minF,maxF,samplerate,
                 seq_len, hidden_size, 
                 attention_size, num_channels):
        super().__init__()
        self.filterbank = LearnableFilterbank(M, num_channels)
        self.BLSTM = BLSTM_Layer(M*num_channels, seq_len, hidden_size)
        
        # Because its bidirectional, feature size is doubled
        self.attention = Attention_Layer(hidden_size, seq_len, attention_size)
    def forward(self, x):
        # Assumes (Epoch, Sequence, Feature)
        #dump(x)
        x = self.filterbank(x)
        #dump(x)
        x = self.BLSTM(x)
        #dump(x)
        
        x = self.attention(x)
        #dump(x)
        return x

class LearnableFilterbank(nn.Module):
    def __init__(self, num_filters, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        filter_shape_generator = FilterbankShape()
        filter_shape = filter_shape_generator.lin_tri_filter_shape(
            nfilt=num_filters,
            nfft=256,
            samplerate=100,
            lowfreq=0,
            highfreq=50,
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
            data = x[:,c,:,:]

            filterbank = torch.multiply(
                torch.sigmoid(self.filter_weights[:,c]),
                self.triangular_matrix,
            ).to(self.device)
            
            data = torch.matmul(data, filterbank)
            chnls.append(data)
        
        # Concatenate the outputs in the M dimension, so that final size is (num_epochs, 29, M*num_channels)
        x = torch.cat(chnls, dim=2)
        
        return x

class Attention_Layer(nn.Module):
    # From Kaare implementation
    def __init__(self, hidden_size, time_bins, attention_size):
        super().__init__()
        self.attweight_w = torch.nn.Parameter(torch.randn(2*hidden_size, attention_size))
        self.attweight_b = torch.nn.Parameter(torch.randn(attention_size))
        self.attweight_u = torch.nn.Parameter(torch.randn(attention_size))
        self.nHidden = hidden_size
        self.time_bins = time_bins
    
    def forward(self, x):
        v = torch.tanh(torch.matmul(torch.reshape(x, [-1, self.nHidden*2]), self.attweight_w) + torch.reshape(self.attweight_b, [1, -1]))
        vu = torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, self.time_bins])
        alphas = exps / torch.reshape(torch.sum(exps, 1), [-1, 1])
        x = torch.sum(x * torch.reshape(alphas, [-1, self.time_bins,1]), 1)
        return x
