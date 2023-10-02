from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt as math_sqrt
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from math import sqrt
from timeit import default_timer as timer

TRAINER_LIB_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    TRAINER_LIB_DEVICE = torch.device("cuda")


def mape(p, t):
    return np.mean(np.abs(p - t) / t)


def mpe(p, t):
    return np.mean((p - t) / t)


def load_country_wide_dataset(file: str):
    df: pd.DataFrame = pd.read_csv(
        file,
        parse_dates=['Time'],
        index_col='Time',
        sep=';'
    )

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year

    df['el_load'] = df['el_load'].clip(
        lower=df['el_load'].quantile(0.001),
        upper=df['el_load'].quantile(0.999)
    )

    df['el_load_lag24'] = df['el_load'].shift(24, fill_value=0)

    return df[['el_load', 'prec', 'grad', 'holiday', 'weekend', 'hour',
               'weekday', 'dayofyear', 'month', 'year', 'el_load_lag24']]


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
    def __init__(self, patience=1, min_delta=0., model: nn.Module | None = None):
        """Passing model is optional, used for checkpointing"""
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = np.inf
        self.__model: nn.Module | None = model
        self.__state_dict = None

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.__model is not None:
                self.__state_dict = deepcopy(self.__model.state_dict())
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_checkpoint(self):
        if self.__model is not None and self.__state_dict is not None:
            with torch.no_grad():
                self.__model.load_state_dict(self.__state_dict)


class Grid:
    def __init__(self, grid: dict):
        self._keys: list = list(grid.keys())
        self._values: list = list(grid.values())
        self._combinations: list = []

    def __iter__(self):
        # choose all combinations of hyperparameters without repetition
        self._combinations = self._values[0]
        if len(self._keys) > 1:
            self._combinations = [[comb] + [item] for item in self._values[1] for comb in self._combinations]
        if len(self._keys) > 2:
            for ls in self._values[2:]:
                self._combinations = [comb + [item] for item in ls for comb in self._combinations]

        return self

    def __next__(self):
        if len(self._combinations) > 0:
            return dict(zip(self._keys, self._combinations.pop(0)))
        else:
            raise StopIteration()


class TSMWrapper(ABC):
    def __init__(self, seq_len: int, pred_len: int):
        self._seq_len: int = seq_len
        self._pred_len: int = pred_len

        self._x_norm_mean: np.ndarray | None = None
        self._x_norm_std: np.ndarray | None = None
        self._y_norm_mean: np.ndarray | None = None
        self._y_norm_std: np.ndarray | None = None

    # region protected methods

    def _std_normalize(self, arr: np.ndarray, which: str, store: bool = False) -> np.ndarray:
        """which can be: 'x' or 'y' or None, if it remains unset, no standard normalization info is stored"""
        if which != 'x' and which != 'y':
            raise ValueError("Argument 'which' can only be: 'x' or 'y'")

        if store:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)

            if which == 'x':
                self._x_norm_mean, self._x_norm_std = mean, std
            else:
                self._y_norm_mean, self._y_norm_std = mean, std
        else:
            if which == 'x':
                if self._x_norm_std is None or self._x_norm_mean is None:
                    raise ValueError("Mean and std for x are not yet stored")

                mean, std = self._x_norm_mean, self._x_norm_std
            else:
                if self._y_norm_std is None or self._y_norm_mean is None:
                    raise ValueError("Mean and std for y are not yet stored")

                mean, std = self._y_norm_mean, self._y_norm_std

        return (arr - mean) / std

    def _std_denormalize(self, arr: np.ndarray, which: str) -> np.ndarray:
        if which != 'x' and which != 'y':
            raise ValueError("Argument 'which' can only be: 'x' or 'y'")

        # checking y here first, since it's more likely to be used
        if which == 'y':
            if self._y_norm_std is None or self._y_norm_mean is None:
                raise ValueError("Mean and std for y are not yet stored")

            mean, std = self._y_norm_mean, self._y_norm_std
        else:
            if self._x_norm_std is None or self._x_norm_mean is None:
                raise ValueError("Mean and std for x are not yet stored")

            mean, std = self._x_norm_mean, self._x_norm_std

        return arr * std + mean

    def _make_ts_dataset(self, x: np.ndarray, y: np.ndarray, store_norm_info: bool = False):
        """Modifies internal mean and std to help denormalization later."""
        x = self._std_normalize(x, 'x', store_norm_info)
        y = self._std_normalize(y, 'y', store_norm_info)

        return TimeSeriesDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)

    # endregion

    # region public methods

    def validate_ts_strategy(self, x: np.ndarray, y: np.ndarray, epochs: int, loss_fn=nn.MSELoss(),
                             lr=0.001, batch_size=128, es_p=10, es_d=0., n_splits=5, verbose=2, cp=True, **kwargs):
        ts_cv = TimeSeriesSplit(n_splits=n_splits)

        train_losses = []
        val_losses = []
        test_losses = []
        metric_losses = []

        st_time = None
        for i, (train_idxs, test_val_idxs) in enumerate(ts_cv.split(x)):
            if verbose > 0:
                st_time = timer()
                print(f"[Fold {i+1}] BEGIN", end="\n" if verbose > 1 else " ")

            test_val_idxs = test_val_idxs[:-len(test_val_idxs)//5]
            test_idxs = test_val_idxs[-len(test_val_idxs)//5:]

            x_train, x_val, x_test = x[train_idxs], x[test_val_idxs], x[test_idxs]
            y_train, y_val, y_test = y[train_idxs], y[test_val_idxs], y[test_idxs]

            self.init_strategy()
            train_loss, val_loss, test_loss = self.train_strategy(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                  epochs=epochs, lr=lr, batch_size=batch_size,
                                                                  loss_fn=loss_fn, es_p=es_p, es_d=es_d,
                                                                  verbose=verbose-1, cp=cp)

            pred, true = self.predict(x_test, y_test)
            metric_loss = sqrt(nn.MSELoss()(torch.tensor(pred), torch.tensor(true)).item())

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            metric_losses.append(metric_loss)

            if verbose > 0:
                elapsed = round((timer() - st_time) / 60, 1)
                print(f"[Fold {i+1}] END" if verbose > 1 else "- END",
                      f"- RMSE loss: {metric_loss:.3f} - Time: {elapsed} min.")

        return train_losses, val_losses, test_losses, metric_losses

    def grid_search(self, x: np.ndarray, y: np.ndarray, g: Grid, loss_fn=nn.MSELoss(), verbose=1):
        best_params = None
        best_score = np.inf
        for i, params in enumerate(g):
            if verbose > 0:
                print(f"[Grid search {i+1:03d}] BEGIN",
                      end=f" - params: {params}\n" if verbose > 1 else " ")

            self._setup_strategy(**params)
            _, _, _, metric_losses = self.validate_ts_strategy(x, y, loss_fn=loss_fn, verbose=verbose-2, **params)

            score = sum(metric_losses) / len(metric_losses)

            improved = score < best_score
            best_params = params if improved else best_params
            best_score = score if improved else best_score

            if verbose > 0:
                print(f"[Grid search {i+1:03d}]" if verbose > 1 else f"-",
                      f"END - Score: {score:.8f} {'*' if improved else ''}")

        return best_params, best_score

    def predict(self, x: np.ndarray, y: np.ndarray):
        self._switch_to_eval_mode()
        dataset: TimeSeriesDataset = self._make_ts_dataset(x, y)
        loader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = np.zeros((0, self._pred_len), dtype=np.float32)
        true = np.zeros((0, self._pred_len), dtype=np.float32)

        with torch.no_grad():
            for features, labels in loader:
                preds = self._predict_strategy(features)
                labels = labels.numpy()
                predictions = np.vstack((predictions, preds))
                true = np.vstack((true, labels))

        return self._std_denormalize(predictions, 'y'), self._std_denormalize(true, 'y')

    # endregion

    # region abstract methods

    @abstractmethod
    def init_strategy(self):
        """
        Used to reset the strategy to its initial state.
        Used in cross validation. Should initialize the model(s).
        """
        pass

    @abstractmethod
    def train_strategy(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, epochs=100, lr=0.001,
                       optimizer=None, batch_size=128, loss_fn=nn.MSELoss(), es_p=10, es_d=0., verbose=1, cp=False):
        pass

    @abstractmethod
    def _setup_strategy(self, **kwargs):
        """
        This method should initialize any parameters that are needed for the strategy to work.
        Training parameters should not be initalized here.
        """
        pass

    @abstractmethod
    def _switch_to_eval_mode(self):
        pass

    @abstractmethod
    def _predict_strategy(self, features) -> np.ndarray:
        pass

    # endregion

    # region static methods

    @staticmethod
    def train_epoch(model: nn.Module, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(model.parameters(), lr=lr)

        model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    test_loader: DataLoader | None = None, epochs=100, lr=0.001, optimizer=None,
                    loss_fn=nn.MSELoss(), es_p=10, es_d=0., verbose=1, cp=False):
        optimizer: torch.optim.Optimizer = optimizer or torch.optim.NAdam(model.parameters(), lr=lr)

        has_test: bool = test_loader is not None

        train_losses: list = []
        val_losses: list = []
        test_losses: list = [] if has_test else None

        early_stopper = EarlyStopper(patience=es_p, min_delta=es_d, model=None if not cp else model)
        for epoch in range(epochs):
            train_loss: float = TSMWrapper.train_epoch(model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn)
            train_losses.append(train_loss)

            val_loss: float = TSMWrapper.test_model(model, val_loader, loss_fn=loss_fn)
            val_losses.append(val_loss)

            test_loss: float | None = None if not has_test else TSMWrapper.test_model(model, test_loader,
                                                                                      loss_fn=loss_fn)
            if test_loss:
                test_losses.append(test_loss)

            text_test_loss: str = "" if not has_test else f", test loss: {test_loss:.6f}"  # to make printing easier

            # stop condition
            if verbose > 0 and epoch > 10 and early_stopper(val_loss):
                print("\r" + " " * 75, end="")
                print(f"\rEarly stopping... Epoch {epoch+1:03d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}{text_test_loss}", end="")
                break

            if verbose > 0:
                print("\r" + " " * 75, end="")
                print(f"\r\tEpoch {epoch+1:03d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}{text_test_loss}", end="")

        if verbose > 0:
            print()  # to get newline after the last epoch
        early_stopper.load_checkpoint()
        return train_losses, val_losses, test_losses

    @staticmethod
    def test_model(model: nn.Module, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        model.eval()
        total_loss: float = 0

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(TRAINER_LIB_DEVICE)
                labels = labels.to(TRAINER_LIB_DEVICE)
                outputs = model(features)

                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def print_evaluation_info(preds: np.ndarray, true: np.ndarray):
        loss = nn.MSELoss()(torch.tensor(preds), torch.tensor(true)).item()
        print("Overall loss metrics:")
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
    def plot_predictions(y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must be equal")

        n_of_plots = y_true.shape[1]
        fig, axs = plt.subplots(nrows=n_of_plots, ncols=1, figsize=(30, 12*n_of_plots))
        if n_of_plots == 1:
            axs = [axs]  # if we have 1 plot, the returned axs is not a list, but a single object
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

    # endregion


class MIMOTSWrapper(TSMWrapper):
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int):
        super(MIMOTSWrapper, self).__init__(seq_len=seq_len, pred_len=pred_len)
        self._model = model.to(TRAINER_LIB_DEVICE)

    # region override methods
    def init_strategy(self):
        self._model.__init__()
        self._model = self._model.to(TRAINER_LIB_DEVICE)

    def _setup_strategy(self, **kwargs):
        if kwargs.get('model', None) is not None:
            self._model = kwargs['model'](**kwargs).to(TRAINER_LIB_DEVICE)

    def train_strategy(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, epochs=100, lr=0.001,
                       optimizer=None, batch_size=128, loss_fn=nn.MSELoss(), es_p=10, es_d=0.,
                       verbose=1, cp=False, **kwargs):

        train_dataset: TimeSeriesDataset = self._make_ts_dataset(x_train, y_train, store_norm_info=True)
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset: TimeSeriesDataset = self._make_ts_dataset(x_val, y_val)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_loader: DataLoader | None = None
        if x_test is not None and y_test is not None:
            test_dataset: TimeSeriesDataset = self._make_ts_dataset(x_test, y_test)
            test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return TSMWrapper.train_model(self._model, train_loader, val_loader, test_loader, epochs=epochs, lr=lr,
                                      optimizer=optimizer, loss_fn=loss_fn, es_p=es_p, es_d=es_d,
                                      verbose=verbose, cp=cp)

    def _switch_to_eval_mode(self):
        self._model.eval()

    def _predict_strategy(self, features: torch.Tensor) -> np.ndarray:
        features = features.to(TRAINER_LIB_DEVICE)
        preds = self._model(features)
        return preds.cpu().numpy()

    # endregion


class GaussianNoise(nn.Module):
    """From: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4"""

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
