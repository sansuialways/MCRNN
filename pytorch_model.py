from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch
import numpy as np


# ------ utils ------ #
def CUDA(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def mse_loss(x_pred, x):
    mse = F.mse_loss(x_pred, x, reduction='mean')
    return mse


def bce_loss(x_pred, x):
    bce = F.binary_cross_entropy(x_pred.contiguous().view(-1, x.size(1)*x.size(2)), x.view(-1, x.size(1)*x.size(2)), reduction='mean')
    return bce

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
# ------ models ------ #

class MCRNN_SED(nn.Module):
    def __init__(self, data_in, data_out):
        super(MCRNN_SED, self).__init__()
        self.rnn_input_size = 128
        self.hidden_size = 64
        self.dropout_rate = 0.2
        self.num_direction = 2
        self.num_layers = 2
        self.data_in = data_in
        self.data_out = data_out
        self.tf = MTF(self.data_in)
        

        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=self.dropout_rate
        )

        self.fc_sed = nn.Sequential(
            nn.Linear(self.hidden_size*self.num_direction, self.hidden_size*2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
            nn.Linear(self.hidden_size*2, data_out[-1]),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*self.num_direction, batch_size, hidden_size)))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(2)
        output=self.tf(x)
        output = output.permute(0, 2, 1, 3)

        output = output.contiguous().view(batch_size,seq_len,128)
        h_state = self.init_hidden(batch_size, self.hidden_size)
        self.rnn.flatten_parameters()
        output, h_state = self.rnn(output, h_state)

        sed = self.fc_sed(output)

        return sed  

class MTF(nn.Module):
    def __init__(self, data_in):
        super(MTF, self).__init__()
        
        self.dropout_rate = 0.2
        self.out1 = 60
        self.out2 = 8
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=data_in[1], out_channels=32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=self.out1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(self.dropout_rate)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=data_in[1], out_channels=32, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=self.out1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.out1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(self.dropout_rate)
        )   
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=data_in[1], out_channels=32, kernel_size=7, stride=1, padding=3),  
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=self.out2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.out2),
            nn.ReLU(True),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(self.dropout_rate)
        ) 

    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*2, batch_size, hidden_size)))

    def forward(self, x):
 
        branch1 = self.cnn1(x)
        branch2 = self.cnn2(x)
        branch3 = self.cnn3(x)

        output = (branch1, branch2, branch3)#branch3,branch4)
        output = torch.cat(output, 1)
        return output  
class CRNN_SED(nn.Module):
    def __init__(self, data_in, data_out):
        super(CRNN_SED, self).__init__()
        self.rnn_input_size = 128
        self.hidden_size = 64
        self.dropout_rate = 0.2
        self.num_direction = 2
        self.num_layers = 2
        self.data_in = data_in
        self.data_out = data_out
        self.maxpooling_channel = [4, 4, 4]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=data_in[1], out_channels=64, kernel_size=3, stride=1, padding=1),  

            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.maxpooling_channel[0])),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.maxpooling_channel[1])),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),    
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.maxpooling_channel[2])),
            nn.Dropout(self.dropout_rate)
        )
            
        # rnn-gru layer
        # [B, 128, 64*2] -> [B, 128, 128]
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=self.dropout_rate
        )

        # fully connected layer for sed
        # [B, 128, 128] -> [B, 128, 22]
        self.fc_sed = nn.Sequential(
            nn.Linear(self.hidden_size*self.num_direction, self.hidden_size*2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
            nn.Linear(self.hidden_size*2, data_out[-1]),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*self.num_direction, batch_size, hidden_size)))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(2)
        
        output = self.cnn(x)
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, seq_len, 128)
        
        h_state = self.init_hidden(batch_size, self.hidden_size)
        self.rnn.flatten_parameters()
        output, h_state = self.rnn(output, h_state)
        
        sed = self.fc_sed(output)

        return sed

 
class CRNN_Baseline(nn.Module):
    def __init__(self, data_in, data_out, doa_type=None, **kwargs):
        super(CRNN_Baseline, self).__init__()
        self.rnn_input_size = 128
        self.hidden_size = 64
        self.dropout_rate = kwargs['dropout_rate']
        self.num_direction = 2
        self.num_layers = 2
        self.data_in = data_in
        self.data_out = data_out
        self.mp_size = 8 if data_in[3] == 1024 else 4
        self.doa_type = doa_type

        # 1d-conv layer
        # [B, C, T, F] = [B, C, T, F] -> [B, 64, T, F/64]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=data_in[1], out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.mp_size)),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.mp_size)),
            nn.Dropout(self.dropout_rate),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d((1, self.mp_size)),
            nn.Dropout(self.dropout_rate)
        )

        # rnn-gru layer
        # [B, 128, 64*2] -> [B, 128, 128]
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=self.dropout_rate
        )

        # fully connected layer for SED
        # [B, T, 128] -> [B, T, 11]
        self.fc_sed = nn.Sequential(
            nn.Linear(self.hidden_size*self.num_direction, self.hidden_size*2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
            nn.Linear(self.hidden_size*2, data_out[0][-1]),
            nn.Sigmoid()
        )

        # fully connected layer for DOA
        if not self.doa_type or self.doa_type == 'reg':
            # [B, T, 128] -> [B, T, 22]
            self.fc_doa = nn.Sequential(
                #nn.Linear(self.hidden_size*self.num_direction, self.hidden_size*2),
                #nn.Dropout(self.dropout_rate),
                #nn.ReLU(True),
                #nn.Linear(self.hidden_size*2, data_out[1][-1])
                nn.Linear(self.hidden_size*self.num_direction, data_out[1][-1]),
            )
        elif self.doa_type == 'cls':
            # [B, T, 128] -> [B, T, 11*(36+9)]
            self.fc_doa = nn.Sequential(
                nn.Linear(self.hidden_size*self.num_direction, 11*data_out[1][-1]),
                nn.Sigmoid()
            )


    def init_hidden(self, batch_size, hidden_size):
        return CUDA(Variable(torch.zeros(self.num_layers*self.num_direction, batch_size, hidden_size)))

    def forward(self, x, **kwargs):
        batch_size, T = x.size(0), x.size(2)
        
        output = self.cnn(x)
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(batch_size, T, 128)

        h_state = self.init_hidden(batch_size, self.hidden_size)
        self.rnn.flatten_parameters()
        output, h_state = self.rnn(output, h_state)

        sed = self.fc_sed(output)
        doa = self.fc_doa(output)

        if self.doa_type == 'cls':
            doa = doa.view(batch_size, T, 11, 36+9)

        return sed, doa

