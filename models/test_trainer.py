"""
SCRIPT TO TEST THE TRAINER LIBRARY, NOT PART OF THE MAIN PROJECT
"""
import numpy as np
import torch
from torch import nn
import trainer_lib


df = trainer_lib.format_country_wide_dataset('../data/final_dataframe.csv')


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


X = df.to_numpy(dtype=np.float32)
y = df['el_load'].to_numpy(dtype=np.float32)

model = Model()
wrap: trainer_lib.TSMWrapper = trainer_lib.TSMWrapper(model, 1, 1)
wrap.validate_ts_model(X, y, epochs=10, lr=0.01)
trainer_lib.TSMWrapper.print_evaluation_info(*wrap.predict_ts_model(X, y))
