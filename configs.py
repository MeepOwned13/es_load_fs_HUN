import torch
from torch import nn
import models.trainer_lib as tl
import models.torch_model_definitions as tmd
from torch.nn.utils import weight_norm
import numpy as np


# region Model definitions

class ConvNetLong(nn.Module):
    def __init__(self, seq_len=48, dropout=0, noise=0.05, **kwargs):
        super(ConvNetLong, self).__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d((8, 0, 0, 0)),
            weight_norm(nn.Conv1d(1, 64, 16)),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=0),
            nn.ZeroPad2d((7, 0, 0, 0)),
            weight_norm(nn.Conv1d(64, 128, 14)),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=0),
        )
        out = self.conv(torch.randn(1, 1, seq_len)).shape[-1]
        self.fc = nn.Sequential(
            tmd.GaussianNoise(noise),
            nn.Flatten(1, -1),
            nn.Linear(128 * out, 2 * 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * 64, 3),
        )
        self.seq_len = seq_len

    def forward(self, x):
        x = x.reshape(-1, 1, self.seq_len)
        x = self.conv(x)
        x = self.fc(x)
        return x


class TCN(nn.Module):
    def __init__(self, seq_len=24, pred_len=3, num_channels=(24,) * 2,
                 kernel_size=3, dropout=0.5, noise=0.0, **kwargs):
        super(TCN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tcn = tmd.TemporalConvNet(1, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.hid_noise = tmd.GaussianNoise(noise)
        self.fc = nn.Linear(num_channels[-1], pred_len)

    def forward(self, x):
        x = x.reshape(-1, 1, self.seq_len)
        x = self.tcn(x)
        x = self.hid_noise(x)
        return self.fc(x[:, :, -1])


class LSTM(nn.Module):
    def __init__(self, features=11, pred_len=3, hidden_size=15, num_layers=2,
                 dropout=0.0, noise=0.0, bidirectional=True, **kwargs):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.h_n_dim = 2 if bidirectional else 1
        self.num_layers = num_layers
        rec_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=features, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=rec_drop)
        self.fc = nn.Sequential(
            nn.Flatten(),
            tmd.GaussianNoise(noise),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.h_n_dim * self.num_layers, pred_len)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.h_n_dim * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(tl.TRAINER_LIB_DEVICE)
        c_0 = torch.zeros(self.h_n_dim * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(tl.TRAINER_LIB_DEVICE)

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = torch.permute(h_n, (1, 0, 2))
        return self.fc(h_n)


class GRU(nn.Module):
    def __init__(self, features=11, pred_len=3, hidden_size=20, num_layers=2,
                 dropout=0.0, noise=0.0, bidirectional=True, **kwargs):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.h_n_dim = 2 if bidirectional else 1
        self.num_layers = num_layers
        rec_drop = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size=features, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=rec_drop)
        self.fc = nn.Sequential(
            nn.Flatten(),
            tmd.GaussianNoise(noise),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.h_n_dim * self.num_layers, features)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        hidden = torch.zeros(self.h_n_dim * self.num_layers, batch_size, self.hidden_size).requires_grad_().to(
            tl.TRAINER_LIB_DEVICE)

        _, hidden = self.gru(x, hidden)
        x = torch.permute(hidden, (1, 0, 2))
        return self.fc(x)


# endregion


# region Configs

CONFIGS = {
    'mimo_cnn': {
        'n_splits': 9,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 1024,
        'es_p': 20,
        'wrapper': tl.MIMOTSWrapper,
        'file_name': 'final_eval_results/mimo_cnn.csv',
        'model': ConvNetLong,
        'model_params': {
            'seq_len': 48,
            'dropout': 0.5,
            'noise': 0.02,
        },
        'seq_len': 48,
        'pred_len': 3,
        'load_modifier': 'only_el_load',
    },
    'mimo_tcn': {
        'n_splits': 9,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 1024,
        'es_p': 10,
        'wrapper': tl.MIMOTSWrapper,
        'file_name': 'final_eval_results/mimo_tcn.csv',
        'model': TCN,
        'model_params': {
            'seq_len': 48,
            'pred_len': 3,
            'num_channels': (72,) * 4,
            'kernel_size': 5,
            'dropout': 0.3,
            'noise': 0.0,
        },
        'seq_len': 48,
        'pred_len': 3,
        'load_modifier': 'only_el_load',
    },
    'mimo_lstm': {
        'n_splits': 9,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.MIMOTSWrapper,
        'file_name': 'final_eval_results/mimo_lstm.csv',
        'model': LSTM,
        'model_params': {
            'features': 11,
            'pred_len': 3,
            'hidden_size': 20,
            'num_layers': 3,
            'bidirectional': True,
            'dropout': 0.3,
            'out_noise': 0.05
        },
        'seq_len': 24,
        'pred_len': 3,
        'load_modifier': 'regular',
    },
    'seq2seq': {
        'n_splits': 9,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.S2STSWRAPPER,
        'file_name': 'final_eval_results/seq2seq.csv',
        'model': tmd.Seq2seq,
        'model_params': {
            'features': 11,
            'pred_len': 3,
            'embedding_size': 10,
            'num_layers': 1,
            'bidirectional': True,
            'dropout': 0.5,
            'out_noise': 0.05
        },
        'seq_len': 24,
        'pred_len': 3,
        'load_modifier': 'regular',
    },
    'rec_om': {
        'n_splits': 9,
        'epochs': 1000,
        'lr': 0.001,
        'batch_size': 2048,
        'es_p': 20,
        'wrapper': tl.RECOneModelTSWrapper,
        'file_name': 'final_eval_results/rec_om.csv',
        'model': GRU,
        'model_params': {
            'features': 11,
            'pred_len': 11,
            'hidden_size': 70,
            'num_layers': 5,
            'bidirectional': True,
            'dropout': 0.5,
            'out_noise': 0.05
        },
        'seq_len': 24,
        'pred_len': 3,
        'load_modifier': 'both_full',
    },

}

# endregion


# region helper funcs

def load_data(load_modifier='regular'):
    dataset = tl.load_country_wide_dataset('data/country_data.csv')
    if load_modifier == 'regular':
        l_x = dataset.to_numpy(dtype=np.float32)
        l_y = dataset['el_load'].to_numpy(dtype=np.float32)
    elif load_modifier == 'only_el_load':
        l_x = dataset['el_load'].to_numpy(dtype=np.float32)
        l_y = l_x.copy()
    elif load_modifier == 'both_full':
        l_x = dataset.to_numpy(dtype=np.float32)
        l_y = l_x.copy()
    else:
        raise ValueError(f"Invalid load modifier: {load_modifier}")
    return l_x, l_y


# endregion
