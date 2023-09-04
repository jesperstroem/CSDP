import torch.nn as nn
import torch
from .bn_blstm import BLSTM_Layer, BLSTM_Layer_Torch
from .basic_layers import FC

class LongSequenceModel(nn.Module):
    class Config():
        def __init__(self, K, B, lstm_input_size, lstm_hidden_size):
            self.K = K
            self.B = B
            self.lstm_input_size = lstm_input_size
            self.lstm_hidden_size = lstm_hidden_size
            
    def __init__(self, config: Config):
        super().__init__()
        self.folder = Folder(config.K,config.B)
        self.unfolder = Unfolder()
        self.intra = SubsequenceModel(config.K,config.B,config.lstm_input_size, config.lstm_hidden_size)
        #Input K and B reverse because the matrix is transposed
        self.inter = SubsequenceModel(config.B,config.K,config.lstm_input_size, config.lstm_hidden_size)
    
    def forward(self, x):
        # x is (Batch, Epoch, Feature)
        
        x = self.folder(x)
        # (Batch, B, K, Feature)
        x = self.intra(x)
        
        #print(x)
        #print(x.shape)
        # Transpose tensor to do inter modeling
        x = torch.transpose(x, 1,2)
        #print(x.shape)
        # (Batch, B, K, Feature)
        x = self.inter(x)
        
        # Transpose again to get the original shape
        x = torch.transpose(x, 1, 2)
        
        # (Batch, B, K, Feature)
        x = self.unfolder(x)
        # (Batch, Epoch, Feature)
        return x

class Folder(nn.Module):
    def __init__(self, K, B):
        super().__init__()       
        self.K = K
        self.B = B
        
    def forward(self, x):
        # Assumes (Batch, Epoch, Features)
        num_batches, num_epochs, num_features = x.shape
        assert(num_epochs==self.K*self.B), f"Num epochs: {num_epochs}, K: {self.K}, B: {self.B}"
        
        x = x.unflatten(1, sizes=(self.B,self.K))
        
        # Returns (Batch, B, K, Features)
        return x
    
class Unfolder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=2)
        return x

class SubsequenceModel(nn.Module):
    def __init__(self, K, B, input_size, hidden_size):
        super().__init__()
        
        self.blstm = BLSTM_Layer(input_size, K, hidden_size)
        self.fc = FC(hidden_size*2, input_size, dropout=0.1, activation="none")
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.B = B
        self.K = K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        # Assumes (Batch, B, K, Features)
        
        num_batch, B, K, num_features = x.shape
        
        assert(self.K == K)
        assert(self.B == B)
        assert(self.input_size == num_features)
        
        # Flatten to (Batch*B, K, Features)
        x = torch.reshape(x, (-1, K, num_features))
        residual = x.to(self.device)
        
        # BLSTM
        x = self.blstm(x)
          
        # FC layer
        # First flatten to (Batch*B*K, Features)
        x = torch.reshape(x, (-1, self.hidden_size*2))
        x = self.fc(x)
        
        # Layer Normalization
        # TODO: Is this correct with the dimension?
        x = torch.layer_norm(x, [self.hidden_size*2])
        #torch.layer_norm()
        
        # Unflatten to (Batch*B, K, Features)
        x = torch.reshape(x, (-1, K, self.hidden_size*2))
                
        # Residual connection
        # TODO: Is this the correct way to do residual?
        x += residual

        # Unflatten to (Batch, B, K, Features)
        x = torch.reshape(x, (-1, B, K, self.hidden_size*2))

        return x
    
    
batch_size = 3

features = 16
epochs = 15

K = 5
B = 3
samplerate = 100

lstm_hidden_size = 8

lsm_conf = LongSequenceModel.Config(K, B, lstm_input_size=lstm_hidden_size*2,
                                    lstm_hidden_size=lstm_hidden_size)

net = LongSequenceModel(lsm_conf)

x = torch.randn(batch_size, epochs, features)

x = net(x)