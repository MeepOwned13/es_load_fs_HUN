"""
SCRIPT TO TEST THE TRAINER LIBRARY, NOT PART OF THE MAIN PROJECT
"""
import numpy as np
import torch
from torch import nn
import trainer_lib


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(11, 25)
        self.fc2 = nn.Linear(25, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


df = trainer_lib.load_country_wide_dataset('../data/country_data.csv')

X = df.to_numpy(dtype=np.float32)
y = df['el_load'].to_numpy(dtype=np.float32)

gi = trainer_lib.Grid({
    'lr': [0.001],
    'epochs': [10],
    'batch_size': [64],
})

model = Model()
wrap: trainer_lib.MIMOTSWrapper = trainer_lib.MIMOTSWrapper(model, 1, 1)
b_p, b_s = wrap.grid_search(X, y, gi)
wrap.init_strategy()
wrap.train_strategy(X[:-5000], y[:-5000], X[-5000:-3000], y[-5000:-3000], X[-3000:], y[-3000:], **b_p)
wrap.print_evaluation_info(*wrap.predict(X[-3000:], y[-3000:]))
