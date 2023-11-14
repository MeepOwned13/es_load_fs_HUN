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
    """
    Loads country_wide dataset from file
    :param file: path to file
    :param nodrop: don't drop any columns
    :return: pandas Dataframe, with added time-series and lagged features
    """
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
    """
    CPU dataset class for time series data
    """
    def __init__(self, x, y, seq_len=5, pred_len=1):
        """
        Initializes the dataset
        :param x: features
        :param y: what to predict
        :param seq_len: sequence length
        :param pred_len: prediction length
        """
        self.X = x
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        """
        Returns the length of the dataset
        :return: length of dataset
        """
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        """
        Returns the item at given index
        :param idx: what index to return
        :return: (features, labels) of correct lengths specified in __init__()
        """
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


class TimeSeriesTensorDataset(TensorDataset):
    """
    GPU dataset class for time series data
    """
    def __init__(self, x, y, seq_len=5, pred_len=1):
        """
        Initializes the dataset, moves tensors to GPU
        :param x: features
        :param y: what to predict
        :param seq_len: sequence length
        :param pred_len: prediction length
        """
        super(TimeSeriesTensorDataset, self).__init__(torch.tensor(x), torch.tensor(y))
        self.X = self.tensors[0].to(TRAINER_LIB_DEVICE)
        self.y = self.tensors[1].to(TRAINER_LIB_DEVICE)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        """
        Returns the length of the dataset
        :return: length of dataset
        """
        return len(self.X) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        """
        Returns the item at given index
        :param idx: what index to return
        :return: (features, labels) of correct lengths specified in __init__()
        """
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


class EarlyStopper:
    """Class implementing early stopping"""
    def __init__(self, patience=1, min_delta=0., model: nn.Module | None = None):
        """
        Initializes the early stopper
        :param patience: how many epochs to wait before stopping
        :param min_delta: what is the minimum delta to not consider as deterioration
        :param model: optional, used for checkpointing
        """
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = np.inf
        self.__model: nn.Module | None = model
        self.__state_dict = None

    def __call__(self, validation_loss):
        """
        Checks if we should stop
        :param validation_loss: current validation loss
        :return: True if we should stop, False otherwise
        """
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
        """
        Loads saved model checkpoint
        :return: None
        """
        if self.__model is not None and self.__state_dict is not None:
            with torch.no_grad():
                self.__model.load_state_dict(self.__state_dict)


class Grid:
    """
    Class implementing grid search using iterator protocol.
    """
    def __init__(self, grid: dict):
        """
        :param grid: dictionary of hyperparameters to search through, specify each key and a list of values
        """
        self._keys: list = list(grid.keys())
        self._values: list = list(grid.values())
        self._combinations: list = []

    def __iter__(self):
        """
        Chooses all combinations of hyperparameters without repetition, not taking order into account
        :return: self
        """
        self._combinations = self._values[0]
        if len(self._keys) > 1:
            self._combinations = [[comb] + [item] for item in self._values[1] for comb in self._combinations]
        if len(self._keys) > 2:
            for ls in self._values[2:]:
                self._combinations = [comb + [item] for item in ls for comb in self._combinations]

        return self

    def __next__(self):
        """
        Returns next combination of hyperparameters
        :return: next combination of hyperparameters as a dict
        """
        if len(self._combinations) > 0:
            return dict(zip(self._keys, self._combinations.pop(0)))
        else:
            raise StopIteration()


class TSMWrapper(ABC):
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int):
        self._seq_len: int = seq_len
        self._pred_len: int = pred_len

        self._x_norm_mean: np.ndarray | None = None
        self._x_norm_std: np.ndarray | None = None
        self._y_norm_mean: np.ndarray | None = None
        self._y_norm_std: np.ndarray | None = None

        self._model = model

    # region magic methods

    def __del__(self):
        """WARNING: deletion of the object will delete the model as well!"""
        del self._model
        if TRAINER_LIB_DEVICE != torch.device('cpu'):
            torch.cuda.empty_cache()

    # endregion

    # region protected methods

    def _std_normalize(self, arr: np.ndarray, which: str, store: bool = False) -> np.ndarray:
        """
        Handles internal normalization and normalization info storing
        :param arr: array to normalize
        :param which: {'x', 'y'}, for what kind of array should normalization be done
        :param store: specifies if normalization info should be stored
        :return: normalized array
        """
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
        """
        Handles internal denormalization
        :param arr: array to denormalize
        :param which: {'x', 'y'}, for what kind of array should normalization be done
        :return: denormalized array
        """
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
        """
        Handles making the internal datasets used in training, validation and testing
        :param x: X array to be used in dataset
        :param y: y array to be used in dataset
        :param store_norm_info: specifies if normalization info should be stored
        :return: normalized TimeSeriesDataset if device is cpu, TimeSeriesTensorDataset if device is cuda
        """
        x = self._std_normalize(x, 'x', store_norm_info)
        y = self._std_normalize(y, 'y', store_norm_info)

        if TRAINER_LIB_DEVICE == torch.device('cpu'):
            return TimeSeriesDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)
        else:
            return TimeSeriesTensorDataset(x, y, seq_len=self._seq_len, pred_len=self._pred_len)

    def _train_epoch(self, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        """
        Trains the internal model for on epoch
        :param data_loader: training dataloader
        :param lr: learning rate
        :param optimizer: optimizer, defaults to NAdam()
        :param loss_fn: loss function, defaults to MSELoss()
        :return: training loss
        """
        optimizer = optimizer or torch.optim.NAdam(self._model.parameters(), lr=lr)

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

    def _train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                     test_loader: DataLoader | None = None, epochs=100, lr=0.001, optimizer=None,
                     loss_fn=nn.MSELoss(), es_p=10, es_d=0., verbose=1, cp=False):
        """
        Trains, validates and tests internal model, prints info to stdout if requested
        :param train_loader: training dataloader
        :param val_loader: validation dataloader
        :param test_loader: testing dataloader
        :param epochs: how many epochs to train for
        :param lr: learning rate
        :param optimizer: optimizer, defaults to NAdam()
        :param loss_fn: loss function, defaults to MSELoss()
        :param es_p: early stop patience
        :param es_d: early stop delta
        :param verbose: verbosity levels, 0=no output, 1=single line updating output
        :param cp: use checkpointing?
        :return: (training losses, validation losses, test losses)
        """
        optimizer: torch.optim.Optimizer = optimizer or torch.optim.NAdam(self._model.parameters(), lr=lr)

        has_test: bool = test_loader is not None

        train_losses: list = []
        val_losses: list = []
        test_losses: list = [] if has_test else None

        early_stopper = EarlyStopper(patience=es_p, min_delta=es_d, model=None if not cp else self._model)
        for epoch in range(epochs):
            train_loss: float = self._train_epoch(train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn)
            train_losses.append(train_loss)

            val_loss: float = self._test_model(val_loader, loss_fn=loss_fn)
            val_losses.append(val_loss)

            test_loss: float | None = None if not has_test else self._test_model(test_loader, loss_fn=loss_fn)
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

    def _test_model(self, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        """
        Tests internal model
        :param data_loader: testing dataloader
        :param loss_fn: loss function, defaults to MSELoss()
        :return: testing loss
        """
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

    def _reset_all_weights(self):
        """
        Recursively resets the models and it's internal Module-s states with random initialization
        :return: None
        """
        @torch.no_grad()
        def weight_reset(m: nn.Module):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        self._model.apply(fn=weight_reset)

    # endregion

    # region public methods

    def validate_ts_strategy(self, x: np.ndarray, y: np.ndarray, epochs: int, loss_fn=nn.MSELoss(), val_mod=8,
                             lr=0.001, batch_size=128, es_p=10, es_d=0., n_splits=6, verbose=2, cp=True, **kwargs):
        """
        Validates internal time-series model by testing it for given amount of folds, with given parameters
        :param x: X to use for training, validation and testing, should be all the data we have
        :param y: y to use for training, validation and testing, should be all the labels we have
        :param epochs: epochs to train for
        :param loss_fn: loss function, defaults to MSELoss()
        :param val_mod: specifies the amount of data used as a validation set, proportional to the first fold
        :param lr: learning rate
        :param batch_size: batch size
        :param es_p: early stop patience
        :param es_d: early stop delta
        :param n_splits: number of splits to validate for
        :param verbose: verbosity level, 0=no output, 1=no training output, 2=all outputs
        :param cp: use checkpointing?
        :param kwargs: used to set parameters from dict
        :return: (train losses, validation losses, test losses, metric losses) as nested lists for each fold
        """
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
            metric_loss = math_sqrt(nn.MSELoss()(torch.tensor(pred), torch.tensor(true)).item())

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
        """
        Performs grid search with given grid, prints to stdout if specified
        :param x: X to use for training, validation and testing, should be all the data we have
        :param y: y to use for training, validation and testing, should be all the labels we have
        :param g: Grid class we specified the parameters in
        :param loss_fn: loss function, defaults to MSELoss()
        :param verbose: verbosity level, 0=no output, 1=output metric per grid, 2=output fold info, 3=all outputs
        :return: (best_params, best_score) in a dict and float respectively
        """
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
        """
        Predict for given data via the internal model
        :param x: X used for prediction
        :param y: y, only used to return proper dimensions proportional to predictions
        :return: (predictions, true values)
        """
        self._model.eval()
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

    def save_state(self, path):
        """
        Saves internal state to path
        :param path: path to save to
        :return: None
        """
        state = {
            'state_dict': self._model.state_dict(),
            'seq_len': self._seq_len,
            'pred_len': self._pred_len,
            'x_norm_mean': self._x_norm_mean,
            'x_norm_std': self._x_norm_std,
            'y_norm_mean': self._y_norm_mean,
            'y_norm_std': self._y_norm_std,
        }
        torch.save(state, path)

    def load_state(self, path):
        """
        Load model from path, make sure it's a file you saved to via the classes method
        :param path: path to load from
        :return: None
        """
        state = torch.load(path, map_location=TRAINER_LIB_DEVICE)

        self._seq_len = state['seq_len']
        self._pred_len = state['pred_len']
        self._x_norm_mean = state['x_norm_mean']
        self._x_norm_std = state['x_norm_std']
        self._y_norm_mean = state['y_norm_mean']
        self._y_norm_std = state['y_norm_std']

        self._model.load_state_dict(state['state_dict'])
        self._model.to(TRAINER_LIB_DEVICE)
        self._model.eval()

    # endregion

    # region abstract methods

    @abstractmethod
    def init_strategy(self):
        """
        Used to reset the strategy to its initial state.
        Used in cross validation. Should initialize the model.
        """
        self._reset_all_weights()
        self._model = self._model.to(TRAINER_LIB_DEVICE)

    @abstractmethod
    def train_strategy(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                       x_test: np.ndarray | None = None, y_test: np.ndarray | None = None, epochs=100, lr=0.001,
                       optimizer=None, batch_size=128, loss_fn=nn.MSELoss(), es_p=10, es_d=0., verbose=1, cp=False):
        """
        Used to train the strategy on the specificied set of data.
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param x_test:
        :param y_test:
        :param epochs: epochs to train for
        :param lr: learning rate
        :param optimizer: optimizer
        :param batch_size: batch size
        :param loss_fn: loss function, defaults to MSELoss()
        :param es_p: early stop patience
        :param es_d: early stop delta
        :param verbose: verbosity level
        :param cp: use checkpointing?
        :return: (train losses, validation losses, test losses, metric losses)
        """
        pass

    @abstractmethod
    def _setup_strategy(self, **kwargs):
        """
        This method should initialize any parameters that are needed for the strategy to work.
        Training parameters should not be initalized here.
        :param kwargs: parameters to set
        """
        pass

    @abstractmethod
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Strategy specific prediction, used to predict for given data.
        :param features: features to predict from
        :param labels: labels, used to return proper dimensions proportional to predictions
        :return: (predictions, true values)
        """
        pass

    # endregion

    # region static methods

    @staticmethod
    def print_evaluation_info(preds: np.ndarray, true: np.ndarray, to_graph: int = 1000):
        """
        Prints evaluation metrics to stdout, and displays given length graph of predictions.
        :param preds: model predictions
        :param true: true values
        :param to_graph: how many points to graph
        :return: None
        """
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

        if to_graph > 0:
            TSMWrapper.plot_predictions_per_hour(preds[-to_graph:], true[-to_graph:])

    @staticmethod
    def plot_predictions_per_hour(y_pred: np.ndarray, y_true: np.ndarray):
        """
        Plots predictions on different graphs per hour
        :param y_pred: predictions
        :param y_true: true values
        :return: None
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must be equal")
        if y_true.ndim != 2:
            raise ValueError("y_true and y_pred must be 2 dimensional")

        n_of_plots = y_true.shape[1]
        fig, axs = plt.subplots(nrows=n_of_plots, ncols=1, figsize=(50, 12*n_of_plots))
        if n_of_plots == 1:
            axs = [axs]  # if we have 1 plot, the returned axs is not a list, but a single object
        for i in range(n_of_plots):
            true = y_true[:, i]
            pred = y_pred[:, i]

            mae = np.abs(pred - true)

            ax = axs[i]
            ax.set_title(f"Predicting ahead by {i+1} hour")
            ax.plot(true, label="true", color="green")
            ax.plot(pred, label="pred", color="red")

            ax2 = ax.twinx()
            ax2.set_ylim(0, np.max(mae) * 5)
            ax2.bar(np.arange(mae.shape[0]), mae, label="mae", color="purple")

            fontsize = 22
            ax.legend(loc="upper right", fontsize=fontsize)
            ax2.legend(loc="lower right", fontsize=fontsize)

            # set font size
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax2.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(fontsize)
        plt.show()

    @staticmethod
    def plot_predictions_together(y_pred: np.ndarray, y_true: np.ndarray):
        """
        Plots predictions on the same graph, make arrays are 2D, use plot_predictions_per_hour() for 1D
        :param y_pred: predictions
        :param y_true: true values
        :return: None
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must be equal")
        if y_true.ndim != 2:
            raise ValueError("y_true and y_pred must be 2 dimensional")

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 12))
        ax.set_title(f"Predictions together")
        ax.plot(y_true[:, 0], label="true", color="green")
        for i in range(y_true.shape[1]):
            color = (1 - 1 / y_true.shape[1] * i, 0, 1 / y_true.shape[1] * i)
            ax.plot(np.arange(i, i + y_true.shape[0]), y_pred[:, i], label=f"pred {i + 1} hour", color=color)

            fontsize = 22
            ax.legend(loc="lower right", fontsize=fontsize)

            # set font size
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
        plt.show()

    @staticmethod
    def plot_losses(train_losses: list, val_losses: list, test_losses: list):
        """
        Plots losses for each fold
        :param train_losses: training losses
        :param val_losses: validational losses
        :param test_losses: testing losses
        :return: None
        """
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
    """
    Wraps the MIMO strategy for Time-series prediction.
    """
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int):
        """
        Initializes the wrapper
        :param model: model to use
        :param seq_len: sequence length to use
        :param pred_len: length of predictions given
        """
        super(MIMOTSWrapper, self).__init__(model=model, seq_len=seq_len, pred_len=pred_len)

    # region override methods
    @override
    def init_strategy(self):
        super().init_strategy()

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

        return self._train_model(train_loader, val_loader, test_loader, epochs=epochs, lr=lr,
                                 optimizer=optimizer, loss_fn=loss_fn, es_p=es_p, es_d=es_d,
                                 verbose=verbose, cp=cp)

    @override
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        features = features.to(TRAINER_LIB_DEVICE)
        preds = self._model(features)
        return preds.cpu().numpy(), labels.cpu().numpy()

    # endregion


class S2STSWRAPPER(MIMOTSWrapper):
    """
    Wraps the sequence-to-sequence strategy for Time-series prediction.
    """
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int, teacher_forcing_decay=0.01):
        """
        Initializes the wrapper
        :param model: model to use
        :param seq_len: sequence length to use
        :param pred_len: length of predictions given
        :param teacher_forcing_decay: how fast teacher forcing should decay
        """
        super(S2STSWRAPPER, self).__init__(model, seq_len, pred_len)
        if pred_len <= 1:
            raise ValueError("pred_len must be greater than 1")
        self.teacher_forcing_ratio = 0.5
        self.teacher_forcing_decay = teacher_forcing_decay

    # region override methods

    @override
    def init_strategy(self):
        """
        Used to reset the strategy to its initial state.
        Used in cross validation. Should initialize the model.
        Resets teacher forcing ratio too.
        """
        super().init_strategy()
        self.teacher_forcing_ratio = 0.5

    @override
    def _train_epoch(self, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(self._model.parameters(), lr=lr)
        self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.teacher_forcing_decay)

        self._model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            optimizer.zero_grad()
            outputs = self._model(features, labels, self.teacher_forcing_ratio)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    # endregion


class RECOneModelTSWrapper(MIMOTSWrapper):
    """
    Wraps single model recursive strategy for Time-series prediction.
    """
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int,  main_feature=0, teacher_forcing_decay=0.02):
        """
        Initializes the wrapper
        :param model: model to use
        :param seq_len: sequence length to use
        :param pred_len: length of predictions given
        :param main_feature: which feature to predict and evaluate for
        :param teacher_forcing_decay: how fast teacher forcing should decay
        """
        super(RECOneModelTSWrapper, self).__init__(model, seq_len=seq_len, pred_len=pred_len)
        self._teacher_forcing = self._og_tf = 1.0
        self._teacher_forcing_decay = teacher_forcing_decay
        self._main_feature = main_feature

    # region override methods

    @override
    def init_strategy(self):
        """
        Used to reset the strategy to its initial state.
        Used in cross validation. Should initialize the model.
        Resets teacher forcing ratio too.
        """
        super().init_strategy()
        self._teacher_forcing = self._og_tf

    @override
    def _train_epoch(self, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(self._model.parameters(), lr=lr)

        self._teacher_forcing = max(0.0, self._teacher_forcing - 0.02)
        self._model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            for i in range(self._pred_len):
                features = features.to(TRAINER_LIB_DEVICE)
                labels = labels.to(TRAINER_LIB_DEVICE)

                optimizer.zero_grad()
                outputs = self._model(features)
                loss = loss_fn(outputs, labels[:, i])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                features = torch.cat((features[:, 1:], labels[:, i].unsqueeze(1)), dim=1)
                if torch.rand(1) > self._teacher_forcing:
                    features[:, -1] = outputs.detach()

        return total_loss / len(data_loader)

    @override
    def _test_model(self, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        self._model.eval()
        total_loss: float = 0

        with torch.no_grad():
            for features, labels in data_loader:
                for i in range(self._pred_len):
                    features = features.to(TRAINER_LIB_DEVICE)
                    labels = labels.to(TRAINER_LIB_DEVICE)

                    outputs = self._model(features)
                    loss = loss_fn(outputs, labels[:, i])

                    total_loss += loss.item()

                    features = torch.cat((features[:, 1:], outputs.unsqueeze(1)), dim=1)
        return total_loss / len(data_loader)

    @override
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        preds = torch.zeros((features.shape[0], self._pred_len))
        for i in range(self._pred_len):
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            outputs = self._model(features)
            preds[:, i] = outputs[:, self._main_feature]

            features = torch.cat((features[:, 1:], outputs.unsqueeze(1)), dim=1)
        return preds.cpu().numpy(), labels[:, :, self._main_feature].cpu().numpy()

    # endregion


class RECMultiModelTSWrapper(MIMOTSWrapper):
    """
    Wraps multi-model recursive strategy for Time-series prediction.
    """
    def __init__(self, model: nn.Module, seq_len: int, pred_len: int, pred_first_n: int, teacher_forcing_decay=0.02):
        """
        Initializes the wrapper
        :param model: model to use
        :param seq_len: sequence length to use
        :param pred_len: length of predictions given
        :param pred_first_n: first how many features the model predicts, rest are pulled from labels
        :param teacher_forcing_decay: how fast teacher forcing should decay
        """
        super(RECMultiModelTSWrapper, self).__init__(model=model, seq_len=seq_len, pred_len=pred_len)
        if pred_len <= 1:
            raise ValueError("pred_len must be greater than 1")
        self._pred_first_n = pred_first_n
        self._teacher_forcing = self._og_tf = 1.0
        self._teacher_forcing_decay = teacher_forcing_decay

    # region override methods

    @override
    def init_strategy(self):
        super().init_strategy()
        self._teacher_forcing = self._og_tf

    @override
    def _train_epoch(self, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
        optimizer = optimizer or torch.optim.NAdam(self._model.parameters(), lr=lr)

        self._teacher_forcing = max(0.0, self._teacher_forcing - self._teacher_forcing_decay)
        self._model.train()
        total_loss: float = 0

        for features, labels in data_loader:
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            optimizer.zero_grad()
            outputs = self._model(features, labels, self._teacher_forcing)
            loss = loss_fn(outputs, labels[:, :, 0])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    @override
    def _test_model(self, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        self._model.eval()
        total_loss: float = 0

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(TRAINER_LIB_DEVICE)
                labels = labels.to(TRAINER_LIB_DEVICE)

                outputs = self._model(features, labels[:, :, self._pred_first_n:])
                loss = loss_fn(outputs, labels[:, :, 0])

                total_loss += loss.item()
        return total_loss / len(data_loader)

    @override
    def _predict_strategy(self, features: torch.Tensor, labels: torch.Tensor):
        preds = torch.zeros((features.shape[0], self._pred_len))
        for i in range(self._pred_len):
            features = features.to(TRAINER_LIB_DEVICE)
            labels = labels.to(TRAINER_LIB_DEVICE)

            preds = self._model(features, labels[:, :, self._pred_first_n:])
        return preds.cpu().numpy(), labels[:, :, 0].cpu().numpy()

    # endregion
