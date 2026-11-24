import torch
from torch import nn

class BaseCNNLSTM(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        layers=[]
        for cin,cout in zip([inputs,16,32,64],[16,32,64,64]):
            conv=nn.Conv1d(cin,cout,kernel_size=3,padding=1)
            nn.init.xavier_uniform_(conv.weight);nn.init.zeros_(conv.bias)
            layers.extend([conv,nn.ReLU(),nn.MaxPool1d(2,ceil_mode=True),nn.Dropout(.3)])
        self.conv=nn.Sequential(*layers)
        self.lstm=nn.LSTM(64,32,num_layers=2,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,x):
        z=self.conv(x.transpose(1,2)).transpose(1,2)
        h,_=self.lstm(z)
        return self.head(h[:,-1]).squeeze(-1)

class ScaledHead(nn.Sequential):
    def __init__(self, original, output_scale):
        super().__init__(*list(original.children()))
        self.output_scale = float(output_scale)

    def forward(self, x):
        return super().forward(x) * self.output_scale

class CNNLSTM(BaseCNNLSTM):
    def __init__(self, inputs):
        super().__init__(inputs)
        self.head=ScaledHead(self.head,125.)
