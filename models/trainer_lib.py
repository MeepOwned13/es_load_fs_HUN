from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt as math_sqrt
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from math import sqrt
from timeit import default_timer as timer
from overrides import override

TRAINER_LIB_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    TRAINER_LIB_DEVICE = torch.device("cuda")


def mape(p, t):
    return np.mean(np.abs(p - t) / t)


def mpe(p, t):
    return np.mean((p - t) / t)


def load_country_wide_dataset(file: str, nodrop=False):
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

    if nodrop:
        return df

    # returns features that were selected by the feature selection algorithm using Random Forest
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


class TimeSeriesTensorDataset(TensorDataset):
    def __init__(self, x, y, seq_len=5, pred_len=1):
        super(TimeSeriesTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))
        self.X = self.tensors[0].to(TRAINER_LIB_DEVICE)
        self.y = self.tensors[1].to(TRAINER_LIB_DEVICE)
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

        # to avoid division by zero, 0 std happens for example when we are looking at a single year
        std = np.where(std == 0, 1.0, std)

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

        try:
            res = arr * std + mean
        except ValueError:
            res = arr * std[:, 0] + mean[:, 0]

        return res

    def _make_ts_dataset(self, x: np.ndarray, y: np.ndarray, store_norm_info: bool = False):
        """Modifies internal mean and std to help denormalization later."""
        x = self._std_normalize(x, 'x', store_norm_info)
        y = self._std_normalize(y, 'y', store_norm_info)

        if TRAINER_LIB_DEVICE == torch.device('cpu'):
            return TimeSeriesDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)
        else:
            return TimeSeriesTensorDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)

    def _train_epoch(self, model: nn.Module, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
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

    def _train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     test_loader: DataLoader | None = None, epochs=100, lr=0.001, optimizer=None,
                     loss_fn=nn.MSELoss(), es_p=10, es_d=0., verbose=1, cp=False):
        optimizer: torch.optim.Optimizer = optimizer or torch.optim.NAdam(model.parameters(), lr=lr)

        has_test: bool = test_loader is not None

        train_losses: list = []
        val_losses: list = []
        test_losses: list = [] if has_test else None

        early_stopper = EarlyStopper(patience=es_p, min_delta=es_d, model=None if not cp else model)
        for epoch in range(epochs):
            train_loss: float = self._train_epoch(model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn)
            train_losses.append(train_loss)

            val_loss: float = self._test_model(model, val_loader, loss_fn=loss_fn)
            val_losses.append(val_loss)

            test_loss: float | None = None if not has_test else self._test_model(model, test_loader, loss_fn=loss_fn)
            if test_loss:
                test_losses.append(test_loss)

            text_test_loss: str = "" if not has_test else f", test loss: {test_loss:.6f}"  # to make printing easier

            # stop condition
            if epoch > 10 and early_stopper(val_loss):
                if verbose > 0:
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

    def _test_model(self, model: nn.Module, data_loader: DataLoader, loss_fn=nn.MSELoss()):
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

    # endregion

    # region public methods

    def validate_ts_strategy(self, x: np.ndarray, y: np.ndarray, epochs: int, loss_fn=nn.MSELoss(), val_mod=8,
                             lr=0.001, batch_size=128, es_p=10, es_d=0., n_splits=6, verbose=2, cp=True, **kwargs):
        ts_cv = TimeSeriesSplit(n_splits=n_splits)

        train_losses = []
        val_losses = []
        test_losses = []
        metric_losses = []

        st_time = None
        for i, (train_idxs, test_idxs) in enumerate(ts_cv.split(x)):
            if verbose > 0:
                st_time = timer()
                print(f"[Fold {i+1}] BEGIN", end="\n" if verbose > 1 else " ")

            train_val_sp: int = len(x) // (n_splits+1) // max(2, val_mod)
            val_idxs = train_idxs[-train_val_sp:]
            train_idxs = train_idxs[:-train_val_sp]

            x_train, x_val, x_test = x[train_idxs], x[val_idxs], x[test_idxs]
            y_train, y_val, y_test = y[train_idxs], y[val_idxs], y[test_idxs]

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
            score_no_1st = sum(metric_losses[1:]) / (len(metric_losses) - 1)

            improved = score < best_score
            best_params = params if improved else best_params
            best_score = score if improved else best_score

            if verbose > 0:
                print(f"[Grid search {i+1:03d}]" if verbose > 1 else f"-",
                      f"END - Score: {score:.8f} {'* ' if improved else ''}Without 1st split: {score_no_1st}")

        return best_params, best_score

    def predict(self, x: np.ndarray, y: np.ndarray):
        self._switch_to_eval_mode()
        dataset: TimeSeriesDataset = self._make_ts_dataset(x, y)
        loader: DataLoader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = np.zeros((0, self._pred_len), dtype=np.float32)
        true = np.zeros((0, self._pred_len), dtype=np.float32)

        with torch.no_grad():
            for features, labels in loader:
                preds, labels = self._predict_strategy(features, labels)
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
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        pass

    # endregion

    # region static methods

    @staticmethod
    def reset_all_weights(model: nn.Module) -> None:
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        model.apply(fn=weight_reset)

    @staticmethod
    def print_evaluation_info(preds: np.ndarray, true: np.ndarray, to_graph: int = 1000):
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

        TSMWrapper.plot_predictions(preds[-to_graph:], true[-to_graph:])

    @staticmethod
    def plot_predictions(y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must be equal")

        n_of_plots = y_true.shape[1]
        fig, axs = plt.subplots(nrows=n_of_plots, ncols=1, figsize=(50, 12*n_of_plots))
        if n_of_plots == 1:
            axs = [axs]  # if we have 1 plot, the returned axs is not a list, but a single object
        for i in range(n_of_plots):
            true = y_true[:, i]
            pred = y_pred[:, i]

            mae = np.abs(pred - true)

            axs[i].set_title(f"Predicting ahead by {i+1} hour")
            axs[i].plot(true, label="true", color="green")
            axs[i].plot(pred, label="pred", color="red")

            ax2 = axs[i].twinx()
            ax2.set_ylim(0, np.max(mae) * 5)
            ax2.bar(np.arange(mae.shape[0]), mae, label="mae", color="purple")

            axs[i].legend(loc="upper right")
            ax2.legend(loc="lower right")
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
    @override
    def init_strategy(self):
        self.reset_all_weights(self._model)
        self._model = self._model.to(TRAINER_LIB_DEVICE)

    @override
    def _setup_strategy(self, **kwargs):
        if TRAINER_LIB_DEVICE != torch.device('cpu'):
            torch.cuda.empty_cache()
        if kwargs.get('model', None) is not None:
            self._model = kwargs['model'](**kwargs).to(TRAINER_LIB_DEVICE)

    @override
    def train_strategy(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, epochs=100, lr=0.001,
                       optimizer=None, batch_size=128, loss_fn=nn.MSELoss(), es_p=10, es_d=0.,
                       verbose=1, cp=False, **kwargs):

        train_dataset: Dataset = self._make_ts_dataset(x_train, y_train, store_norm_info=True)
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset: Dataset = self._make_ts_dataset(x_val, y_val)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_loader: DataLoader | None = None
        if x_test is not None and y_test is not None:
            test_dataset: Dataset = self._make_ts_dataset(x_test, y_test)
            test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return self._train_model(self._model, train_loader, val_loader, test_loader, epochs=epochs, lr=lr,
                                 optimizer=optimizer, loss_fn=loss_fn, es_p=es_p, es_d=es_d,
                                 verbose=verbose, cp=cp)

    @override
    def _switch_to_eval_mode(self):
        self._model.eval()

    @override
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        features = features.to(TRAINER_LIB_DEVICE)
        preds = self._model(features)
        return preds.cpu().numpy(), labels.cpu().numpy()

    # endregion


class S2STSWRAPPER(MIMOTSWrapper):
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int, teacher_forcing_decay=0.01):
        super(S2STSWRAPPER, self).__init__(model, seq_len, pred_len)
        self.teacher_forcing_ratio = 0.5
        self.teacher_forcing_decay = teacher_forcing_decay

    # region override methods

    @override
    def init_strategy(self):
        super().init_strategy()
        self.teacher_forcing_ratio = 0.5

    @override
    def _train_epoch(self, model: nn.Module, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(model.parameters(), lr=lr)
        self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.teacher_forcing_decay)

        model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            optimizer.zero_grad()
            outputs = model(features, labels, self.teacher_forcing_ratio)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    # endregion


class RECTSWrapper(TSMWrapper):
    def __init__(self, seq_len: int, pred_len: int, pred_features: tuple, config: dict, teacher_forcing_decay=0.02):
        super(RECTSWrapper, self).__init__(seq_len=seq_len, pred_len=pred_len)
        self._idxs_features_to_predict = pred_features  # example: [0, 1, 2]
        self._teacher_forcing = self._og_tf = 1.0
        self._teacher_forcing_decay = teacher_forcing_decay
        # first feature present will be assumed to be the target feature
        self._config = config
        """
        Example: config:
        self._config = {
            0: {
                'model': 'mod_LSTM',  # class
                'use_features': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # list of indexes
            },
            1: {
                'model': 'mod_CNN',
                'use_features': [1]
            },
            2: {
                'model': 'mod_CNN',
                'use_features': [2]
            }
        }
        """
        for k in pred_features:
            if k not in self._config:
                raise ValueError("Key in pred_features must be in config")

        for k in self._config:
            if k not in pred_features:
                raise ValueError("Key in config must be in pred_features")
            self._config[k]['model'] = self._config[k]['model'].to(TRAINER_LIB_DEVICE)

    # region override methods

    @override
    def init_strategy(self):
        self._teacher_forcing = self._og_tf
        for k in self._config:
            self.reset_all_weights(self._config[k]['model'])
            self._config[k]['model'] = self._config[k]['model'].to(TRAINER_LIB_DEVICE)

    @override
    def _setup_strategy(self, **kwargs):
        if TRAINER_LIB_DEVICE != torch.device('cpu'):
            torch.cuda.empty_cache()
        self._config = kwargs['config']

        for k in self._idxs_features_to_predict:
            if k not in self._config:
                raise ValueError("Key in pred_features must be in config")

        for k in self._config:
            if k not in self._idxs_features_to_predict:
                raise ValueError("Key in config must be in pred_features")
            self._config[k]['model'] = self._config[k]['model'].to(TRAINER_LIB_DEVICE)

    @override
    def train_strategy(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, epochs=100, lr=0.001,
                       optimizer=None, batch_size=128, loss_fn=nn.MSELoss(), es_p=10, es_d=0.,
                       verbose=1, cp=False, **kwargs):
        results = {}
        keys_sorted = list(self._config.keys())
        keys_sorted.sort(reverse=True)  # I want to go backwards, to train the most important model last
        # which is the one that predicts the target feature, I want to give it values from the other models too
        for k in keys_sorted:
            feat = self._config[k]['use_features']
            train_dataset: TimeSeriesDataset = self._make_ts_dataset(x_train[:, feat], y_train[:, feat],
                                                                     store_norm_info=True)
            train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            val_dataset: TimeSeriesDataset = self._make_ts_dataset(x_val[:, feat], y_val[:, feat])
            val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            test_loader: DataLoader | None = None
            if x_test is not None and y_test is not None:
                test_dataset: TimeSeriesDataset = self._make_ts_dataset(x_test[:, feat], y_test[:, feat])
                test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(self._config[k]['model'])
            results[k] = self._train_model(self._config[k]['model'], train_loader, val_loader, test_loader,
                                           epochs=epochs, lr=lr, optimizer=optimizer, loss_fn=loss_fn,
                                           es_p=es_p, es_d=es_d, verbose=verbose, cp=cp)
        return results[0]

    @override
    def _train_epoch(self, model: nn.Module, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(model.parameters(), lr=lr)

        # a crude search, but shouldn't affect performance too much here
        idx, config = self._find_config(model)
        # other models don't need to predict the last value
        pred_len = self._pred_len if idx == 0 else self._pred_len - 1

        self._teacher_forcing = max(0.0, self._teacher_forcing - 0.02)
        model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            for i in range(pred_len):
                features = features.to(TRAINER_LIB_DEVICE)
                labels = labels.to(TRAINER_LIB_DEVICE)

                optimizer.zero_grad()
                outputs = model(features)
                if idx != 0:
                    loss = loss_fn(outputs, labels[:, i])
                else:
                    loss = loss_fn(outputs, labels[:, i, idx].unsqueeze(1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if idx != 0:
                    to_concat = outputs.detach().reshape(-1, 1, 1)
                    if torch.rand(1) < self._teacher_forcing:
                        to_concat = labels[:, i].reshape(-1, 1, 1)
                    features = torch.cat((features[:, 1:], to_concat), dim=1)
                else:
                    outs = torch.zeros((features.shape[0], len(self._idxs_features_to_predict))).to(TRAINER_LIB_DEVICE)
                    with torch.no_grad():
                        for k in self._config:
                            if k == idx:
                                continue
                            feat = self._config[k]['use_features']
                            outs[:, k] = self._config[k]['model'](features[:, :, feat]).reshape(-1)
                    outs[:, 0] = outputs.reshape(-1)

                    features = torch.cat((features[:, 1:], labels[:, i, :].unsqueeze(1)), dim=1)
                    if torch.rand(1) > self._teacher_forcing:
                        features[:, -1, self._idxs_features_to_predict] = outs.detach()

        self._teacher_forcing = self._og_tf
        return total_loss / len(data_loader)

    @override
    def _test_model(self, model: nn.Module, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        # a crude search, but shouldn't affect performance too much here
        idx, config = self._find_config(model)
        pred_len = self._pred_len if idx == 0 else self._pred_len - 1
        # other models don't need to predict the last value

        model.eval()
        total_loss: float = 0

        with torch.no_grad():
            for features, labels in data_loader:
                for i in range(pred_len):
                    features = features.to(TRAINER_LIB_DEVICE)
                    labels = labels.to(TRAINER_LIB_DEVICE)

                    outputs = model(features)
                    if len(config['use_features']) == 1:
                        loss = loss_fn(outputs, labels[:, i])
                    else:
                        loss = loss_fn(outputs, labels[:, i, idx].unsqueeze(1))

                    total_loss += loss.item()

                    if len(config['use_features']) == 1:
                        to_concat = outputs.detach().reshape(-1, 1, 1)
                        features = torch.cat((features[:, 1:], to_concat), dim=1)
                    else:
                        outs = torch.zeros((features.shape[0], len(self._idxs_features_to_predict))).to(TRAINER_LIB_DEVICE)
                        for k in self._config:
                            if k == idx:
                                continue
                            feat = self._config[k]['use_features']
                            outs[:, k] = self._config[k]['model'](features[:, :, feat]).reshape(-1)
                        outs[:, 0] = outputs.reshape(-1)

                        features = torch.cat((features[:, 1:], labels[:, i, :].unsqueeze(1)), dim=1)
                        features[:, -1, self._idxs_features_to_predict] = outs.detach()
        return total_loss / len(data_loader)

    @override
    def _switch_to_eval_mode(self):
        for k in self._config:
            self._config[k]['model'].eval()

    @override
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        preds = torch.zeros((features.shape[0], self._pred_len))
        for i in range(self._pred_len):
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            outputs = torch.zeros((features.shape[0], len(self._idxs_features_to_predict))).to(TRAINER_LIB_DEVICE)
            for k in self._config:
                feat = self._config[k]['use_features']
                outputs[:, k] = self._config[k]['model'](features[:, :, feat]).reshape(-1)
            preds[:, i] = outputs[:, 0]

            features = torch.cat((features[:, 1:], labels[:, i, :].unsqueeze(1)), dim=1)
            features[:, -1, self._idxs_features_to_predict] = outputs.detach()
        return preds.cpu().numpy(), labels[:, :, 0].cpu().numpy()

    # endregion

    # region protected methods

    def _find_config(self, model: nn.Module):
        idx = None
        _config = None
        for k in self._config:
            if self._config[k]['model'] is model:
                idx = k
                _config = self._config[k]
                break
        if idx is None or _config is None:
            raise ValueError("Model not found in config")

        return idx, _config

    # endregion
