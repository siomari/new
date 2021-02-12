import torch
import torch.nn as nn 


class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(LSTM_CNN, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers = self.num_layers)
        self.conv = nn.Conv2d(64, 16, 3)


    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        out = self.conv(lstm_output)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        residual1 = x
        x = self.block1(x)
        x += residual1
        x = self.pool(x)

        residual2 = x
        x = self.block2(x)
        x += residual2
        x = self.pool(x)

        residual3 = x
        x = self.block3(x)
        x += residual3
        x = self.pool(x)

        residual4 = x
        x = self.block4(x)
        x += residual4
        x = self.pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self, modelA, modelB):
        super(Decoder, self).__init__()

        self.modelA = modelA
        self.modelB = modelB
        self.conv1 = nn.Conv2d(16, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.softmax = nn.Softmax()

    def forward(self, x1, x2):

        x1 = self.modelA(x1)
        x2 = self.modelB(x2)

        x = torch.cat((x1, x2), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.softmax(x)
        return x

