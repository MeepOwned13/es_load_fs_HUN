import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

MODEL_DEFINITION_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    MODEL_DEFINITION_DEVICE = torch.device("cuda")


# region GaussianNoise, from: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            noise = self.noise.expand(*x.size()).float().normal_() * scale
            return x + noise
        return x


# endregion


# region TemporalConvolution, adapted from: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
#                             and: https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.pad1 = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        self.pad2 = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.pad1, self.conv1, self.relu1, self.dropout1,
                                 self.pad2, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# endregion


# region Time-series Encoder-Decoder

class GRUEncoder(nn.Module):
    def __init__(self, features, embedding_size, num_layers=1, bidirectional=False, dropout=0.0, noise=0.0):
        super(GRUEncoder, self).__init__()
        self.hidden_size = embedding_size
        self.num_layers = num_layers
        self.h_n_dim = 2 if bidirectional else 1
        self.noise = GaussianNoise(0.0)
        self.gru = nn.GRU(features, embedding_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.noise(x)
        h_0 = (torch.zeros(self.num_layers * self.h_n_dim, batch_size, self.hidden_size)
               .requires_grad_().to(MODEL_DEFINITION_DEVICE))
        _, hidden = self.gru(x, h_0)

        return hidden


class GRUDecoder(nn.Module):
    def __init__(self, features, embedded_size, num_layers=1, bidirectional=False, dropout=0.0, noise=0.0):
        super(GRUDecoder, self).__init__()
        self.h_n_dim = 2 if bidirectional else 1
        self.gru = nn.GRU(features, embedded_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.flatten = nn.Flatten(1, -1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedded_size * num_layers * self.h_n_dim, 1)
        self.noise = GaussianNoise(noise)

    def forward(self, x, h):
        x, hidden = self.gru(x, h)

        out = hidden.permute(1, 0, 2)
        out = self.flatten(out)
        out = self.noise(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out, x, hidden


class Seq2seq(nn.Module):
    def __init__(self, features=11, pred_len=3, embedding_size=64, num_layers=1, bidirectional=False,
                 dropout=0.2, in_noise=0.0, out_noise=0.0, **kwargs):
        super(Seq2seq, self).__init__()
        self.pred_len = pred_len
        self.features = features
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.enc = GRUEncoder(features, embedding_size, num_layers, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.0, noise=in_noise)
        self.dec = GRUDecoder(1, embedding_size, num_layers, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.0, noise=out_noise)

    def forward(self, x, y=None, teacher_forcing=0.0):
        batch_size = x.shape[0]
        hidden = self.enc(x)
        dec_input = x[:, -1, 0].reshape(-1, 1, 1)  # this will be y_prev in my case
        output = torch.zeros(batch_size, self.pred_len).to(MODEL_DEFINITION_DEVICE)

        for i in range(self.pred_len):
            out, _, _ = self.dec(dec_input, hidden)
            output[:, i] = out[:, 0]
            if y is not None and torch.rand(1) < teacher_forcing:
                dec_input = torch.cat((dec_input, y[:, i].reshape(-1, 1, 1)), dim=1)
            else:
                dec_input = torch.cat((dec_input, out.unsqueeze(1)), dim=1)

        return output

# endregion
