import math 
import torch.nn.functional as F
import numpy as np
from torch import nn
import torch 


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# class LSTM_CNN(nn.Module):
#     # def __init__(self, input_size, hidden_dim):
#     def __init__(self):
#         super(LSTM_CNN, self).__init__()

#         # self.input_size = input_size
#         # self.hidden_dim = hidden_dim
#         # self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size = , hidden_size = 64, num_layers = 2)
#         self.conv = nn.Conv2d(64, 16, 3)

#     def forward(self, x):
#         h0 = torch.zeros(2, x.size(0), 64)
#         c0 = torch.zeros(2, x.size(0), 64)
#         lstm_output, _ = self.lstm(x)
#         out = self.conv(lstm_output)
#         return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = Conv2dSamePadding(3, 32, 3)
        
        self.conv_last = Conv2dSamePadding(256, 16, 3)
        
        self.block1 = nn.Sequential(
            Conv2dSamePadding(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            Conv2dSamePadding(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.block3 = nn.Sequential(
            Conv2dSamePadding(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.block4 = nn.Sequential(
            Conv2dSamePadding(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(2, 2)
            
    def forward(self, x):
        x = self.conv(x)
        residual1 = x
        x = self.block1(x)
        x += residual1
        x = self.pool(x)
        
        x = self.block2(x)
        x = self.pool(x)
        
        x = self.block3(x)       
        x = self.pool(x)

        x = self.block4(x)      
        x = self.pool(x)
        
        x = self.conv_last(x)
        return x 

class Decoder(nn.Module):
    def __init__(self, encoder):
        super(Decoder, self).__init__()

        self.encoder = Encoder()
        self.conv1 = Conv2dSamePadding(16, 64, 3)
        
        self.conv2 = Conv2dSamePadding(64, 16, 3)
        
        self.conv_out = Conv2dSamePadding(16, 1, 3)
        
        self.tconv1 = nn.ConvTranspose2d(16, 64, kernel_size=6, stride = 4, padding = 1)
        
        self.tconv2 = nn.ConvTranspose2d(64, 16, kernel_size = 8, stride = 4, padding = 2)


    def forward(self, x1):

        x1 = self.encoder(x1)

        # x = F.interpolate(x1, size=(64,64))
        # x = self.conv1(x)
        
        # x = F.interpolate(x, size=(256,256))
        # x = self.conv2(x)
        

        x = self.tconv1(x1)

        x = self.tconv2(x)
                
        x   = self.conv_out(x) 
        # print(x)
        
        # x = nn.Sigmoid()(x)
        
        return x   #  == (b, 1, 256, 256)

