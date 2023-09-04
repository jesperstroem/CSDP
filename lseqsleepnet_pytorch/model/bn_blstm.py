import torch.nn as nn
import torch

#TODO
class BN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Assumes (Batch, values)
        
        
        return x

class BLSTM_Layer(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blstm = BLSTM_Layer_Torch(input_size, seq_len, hidden_size).to(self.device)
        #self.blstm = BLSTM_Layer_Own(input_size, seq_len, hidden_size)
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.blstm(x)
        return x
    
class BN_LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, gamma=0.1, decay=0.95):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.decay = decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_norm = BN()
        
        # Weights and biases with linear layers
        self.x2h = nn.Linear(input_size, 4*hidden_size, bias=True).to(self.device)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=True).to(self.device)

    def forward(self, x, state):
        x = x.to(self.device)
        
        # Batchnumber, values
        assert(x.dim()==2)
        
        c, h = state
        
        assert(x.size(1)==self.input_size), f"x size: {x.size(1)}, input_size: {self.input_size}"
        
        # Calculate the intermediate values
        xh = self.x2h(x)
        hh = self.h2h(h)
        
        # Batch normalization
        bn_xh = self.batch_norm(xh)
        bn_hh = self.batch_norm(hh)
        
        #Concatenate previous hidden and input
        fiog = bn_xh + bn_hh
        
        #Divide into gates
        f,i,o,g = fiog.chunk(4, 1)
        
        #Based on gates, calculate new c
        c_new = (torch.sigmoid(f) * c) + (torch.sigmoid(i) * torch.tanh(g)) 
        
        #Batch normalization
        bn_c_new = self.batch_norm(c_new)
        
        #Calculate new hidden
        h_new = torch.sigmoid(o) * torch.tanh(bn_c_new)
        
        return (h_new, c_new)

#Own with BN
class BLSTM_Layer_Own(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size):
        super().__init__()
        self.fwd = LSTM(input_size, seq_len, hidden_size)
        self.bwd = LSTM(input_size, seq_len, hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x):
        #TODO: Introduce dropout layers?
        (fwd_outputs, _) = self.fwd(x)
        
        # Flip input sequence for the backward layer
        (bwd_outputs,_) = self.bwd(torch.flip(x,(1,)))
        
        # Flip back the backward output
        bwd_outputs = torch.flip(bwd_outputs,(1,))
        
        x = torch.cat((fwd_outputs,bwd_outputs), 2)
        
        x = self.dropout(x)
        
        return x

# Regular LSTM from pytorch
class BLSTM_Layer_Torch(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size):
        super().__init__()
        
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                  batch_first = True, bidirectional = True)
        
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, x):
        #print(x.shape)
        x,_ = self.lstm(x)
        #print(x.shape)
        x = self.dropout(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 gamma=0.1,
                 decay=0.95):
        # Assumes 1 layer
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_len = seq_len
        self.cell = BN_LSTM_Cell(input_size, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        #Assumes (Batch, Sequence, features)
        num_batches, num_sequences, _ = x.size()
        
        h = torch.zeros(num_batches, self.hidden_size).to(self.device)
        c = torch.zeros(num_batches, self.hidden_size).to(self.device)
        
        hidden_outputs = []
        
        for seq in range(num_sequences):
            xseq = x[:,seq,:]
            
            h, c = self.cell(xseq, (h,c))
            
            # Add sequence dimension
            hidden_outputs.append(h.unsqueeze(1))
        
        # Cat on sequence dimension to get output of (batch, sequence, features)
        return torch.cat(hidden_outputs, dim=1), (h,c)