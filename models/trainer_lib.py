import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt as math_sqrt
import matplotlib.pyplot as plt

TRAINER_LIB_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    TRAINER_LIB_DEVICE = torch.device("cuda")


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, seq_len=5, pred_len=1):
        self.X = x
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TSMWrapper:
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int):
        self._model: nn.Module = model.to(TRAINER_LIB_DEVICE)
        self._seq_len: int = seq_len
        self._pred_len: int = pred_len

        self._x_norm_mean: np.ndarray | None = None
        self._x_norm_std: np.ndarray | None = None
        self._y_norm_mean: np.ndarray | None = None
        self._y_norm_std: np.ndarray | None = None

    def _std_normalize(self, arr: np.ndarray, which: str | None = None) -> np.ndarray:
        """which can be: 'x' or 'y' or None, if it remains unset, no standard normalization info is stored"""
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)

        if which == 'x':
            self._x_norm_mean, self._x_norm_std = mean, std
        elif which == 'y':
            self._y_norm_mean, self._y_norm_std = mean, std

        return (arr - mean) / std

    def _std_denormalize(self, arr: np.ndarray, which: str) -> np.ndarray:
        if which == 'x':
            mean, std = self._x_norm_mean, self._x_norm_std
        elif which == 'y':
            mean, std = self._y_norm_mean, self._y_norm_std
        else:
            raise ValueError("Argument 'which' can only be: 'x' or 'y'")

        return arr * std + mean

    def _make_ts_dataset(self, x: np.ndarray, y: np.ndarray, store_norm_info: bool = False):
        """Modifies internal mean and std to help denormalization later."""
        st_x, st_y = None, None
        if store_norm_info:
            st_x, st_y = 'x', 'y'
        x = self._std_normalize(x, st_x)
        y = self._std_normalize(y, st_y)

        return TimeSeriesDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)

    def train_epoch(self, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            optimizer.zero_grad()
            outputs = self._model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def test_model(self, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        self._model.eval()
        total_loss: float = 0

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(TRAINER_LIB_DEVICE)
                labels = labels.to(TRAINER_LIB_DEVICE)
                outputs = self._model(features)

                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader | None,
              epochs=100, lr=0.001, optimizer=None, loss_fn=nn.MSELoss(), es_p=10, es_d=0.):
        optimizer: torch.optim.Optimizer = optimizer or torch.optim.Adam(self._model.parameters(), lr=lr)

        has_test: bool = test_loader is not None

        train_losses: list = []
        val_losses: list = []
        test_losses: list = [] if has_test else None

        early_stopper = EarlyStopper(patience=es_p, min_delta=es_d)
        for epoch in range(epochs):
            train_loss: float = self.train_epoch(train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn)
            train_losses.append(train_loss)

            val_loss: float = self.test_model(val_loader, loss_fn=loss_fn)
            val_losses.append(val_loss)

            test_loss: float | None = None if not has_test else self.test_model(test_loader, loss_fn=loss_fn)
            if test_loss:
                test_losses.append(test_loss)

            text_test_loss: str = "" if not has_test else f", test loss: {test_loss:.6f}"  # to make printing shorter

            # stop condition
            if epoch > 10 and early_stopper(val_loss):
                print("\r" + " " * 75, end="")
                print(f"\rEarly stopping...\n\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}{text_test_loss}")
                break

            if test_loader:
                print("\r" + " " * 75, end="")
                print(f"\r\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}{text_test_loss}", end="")
        return train_losses, val_losses, test_losses

    def validate_ts_model(self, x: np.ndarray, y: np.ndarray, epochs: int, loss_fn=nn.MSELoss(),
                          lr=0.001, es_p=10, es_d=0., n_splits=5):
        ts_cv = TimeSeriesSplit(n_splits=n_splits)

        train_losses = []
        val_losses = []
        test_losses = []

        for i, (train_idxs, test_val_idxs) in enumerate(ts_cv.split(x)):
            print(f"[Fold {i+1}] BEGIN")
            self._model.__init__()

            test_val_idxs = test_val_idxs[:-len(test_val_idxs)//5]
            test_idxs = test_val_idxs[-len(test_val_idxs)//5:]

            x_train, x_val, x_test = x[train_idxs], x[test_val_idxs], x[test_idxs]
            y_train, y_val, y_test = y[train_idxs], y[test_val_idxs], y[test_idxs]

            train_dataset: TimeSeriesDataset = self._make_ts_dataset(x_train, y_train, store_norm_info=True)
            val_dataset: TimeSeriesDataset = self._make_ts_dataset(x_val, y_val)
            test_dataset: TimeSeriesDataset = self._make_ts_dataset(x_test, y_test)

            train_loader: DataLoader = DataLoader(train_dataset, batch_size=128, shuffle=False)
            val_loader: DataLoader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            test_loader: DataLoader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            train_loss, test_loss, val_loss = self.train(train_loader, val_loader, test_loader, epochs=epochs,
                                                         lr=lr, loss_fn=loss_fn, es_p=es_p, es_d=es_d)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

            print(f"[Fold {i+1}] END")

        return train_losses, val_losses, test_losses

    def predict_ts_model(self, x: np.ndarray, y: np.ndarray):
        self._model.eval()
        dataset: TimeSeriesDataset = self._make_ts_dataset(x, y)
        loader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = np.zeros((0, self._seq_len), dtype=np.float32)
        true = np.zeros((0, self._seq_len), dtype=np.float32)

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(TRAINER_LIB_DEVICE)
                preds = self._model(features)

                preds = preds.cpu().numpy()
                labels = labels.numpy()
                predictions = np.vstack((predictions, preds))
                true = np.vstack((true, labels))

        return self._std_denormalize(predictions, 'y'), self._std_denormalize(true, 'y')

    @staticmethod
    def print_evaluation_info(preds: np.ndarray, true: np.ndarray):
        def mape(p, t):
            return np.mean(np.abs(p - t) / t)

        def mpe(p, t):
            return np.mean((p - t) / t)

        loss = nn.MSELoss()(torch.tensor(preds), torch.tensor(true)).item()
        print("Losses after denormalization:")
        print(f"MAE: {np.mean(np.abs(preds - true)):8.4f}")
        print(f"MSE: {loss:10.4f}")
        print(f"RMSE: {math_sqrt(loss):8.4f}")
        print(f"MAPE: {mape(preds, true) * 100:6.3f}%")
        print(f"MPE: {mpe(preds, true) * 100:6.3f}%")

        for i in range(preds.shape[1]):
            loss = nn.MSELoss()(torch.tensor(preds[:, i]), torch.tensor(true[:, i])).item()
            print(f"{i+1:2d} hour ahead: "
                  f"MAE: {np.mean(np.abs(preds[:, i] - true[:, i])):8.4f}, ",
                  f"MSE: {loss:10.4f}, "
                  f"RMSE: {math_sqrt(loss):8.4f}, "
                  f"MAPE: {mape(preds[:, i], true[:, i]) * 100:6.3f}%, "
                  f"MPE: {mpe(preds[:, i], true[:, i]) * 100:6.3f}%")

        TSMWrapper.plot_predictions(preds[-1000:], true[-1000:])

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must be equal")

        n_of_plots = y_true.shape[1]
        fig, axs = plt.subplots(nrows=n_of_plots, ncols=1, figsize=(30, 12*n_of_plots))
        for i in range(n_of_plots):
            axs[i].set_title(f"Predicting ahead by {i+1} hour")
            axs[i].plot(y_true[:, i], label="true", color="g")
            axs[i].plot(y_pred[:, i], label="pred", color="r")
            axs[i].legend()
        plt.show()

    @staticmethod
    def plot_losses(train_losses: list, val_losses: list, test_losses: list):
        fig, axs = plt.subplots(nrows=1, ncols=len(train_losses), figsize=(8 * len(train_losses), 7))
        for i in range(len(train_losses)):
            axs[i].set_title(f"{i+1} fold")
            axs[i].plot(train_losses[i], label="train_loss", color="g")
            axs[i].plot(val_losses[i], label="val_loss", color="b")
            axs[i].plot(test_losses[i], label="test_loss", color="r")
            axs[i].legend()
        plt.show()

