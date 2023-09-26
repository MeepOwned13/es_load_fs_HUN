import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt as math_sqrt
import matplotlib.pyplot as plt

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU is available')


def train_epoch(model: nn.Module, data_loader: DataLoader, lr=0.001, optimizer=None, loss_fn=nn.MSELoss()):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    for features, labels in data_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def test_model(model: nn.Module, data_loader: DataLoader, loss_fn=nn.MSELoss()):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader | None,
          epochs=100, lr=0.001, optimizer=None, loss_fn=nn.MSELoss(), es_p=10, es_d=0.):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    test_losses = [] if test_loader else None
    early_stopper = EarlyStopper(patience=es_p, min_delta=es_d)
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn)
        train_losses.append(train_loss)
        val_loss = test_model(model, val_loader, loss_fn=loss_fn)
        val_losses.append(val_loss)

        test_loss = None
        if test_loader:
            test_loss = test_model(model, test_loader, loss_fn=loss_fn)
            test_losses.append(test_loss)

        # stop condition
        if epoch > 10 and early_stopper(val_loss):
            if test_loader:
                print(f"Early stopping...\n\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}, test loss: {test_loss:.6f}")
            else:
                print(f"Early stopping...\n\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}")
            break

        if (epoch+1) % 5 == 0 or epoch == 0:
            if test_loader:
                print(f"\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}, test loss: {test_loss:.6f}")
            else:
                print(f"\tEpoch {epoch+1:3d}: train loss: {train_loss:.6f}, "
                      f"val loss: {val_loss:.6f}")
    return train_losses, val_losses, test_losses


def validate_ts_model(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, seq_len: int, pred_len: int,
                      loss_fn=nn.MSELoss(), lr=0.001, es_p=10, es_d=0., n_splits=5):
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    train_losses = []
    val_losses = []
    test_losses = []
    cross_val_losses = []
    for i, (train_idxs, val_idxs) in enumerate(ts_cv.split(x)):
        print(f"Fold {i+1}:")
        model.__init__()
        model.to(device)
        x_cross_val = x[val_idxs]
        y_cross_val = y[val_idxs]
        val_idxs = val_idxs[:-len(val_idxs)//5]
        test_idxs = val_idxs[-len(val_idxs)//5:]
        x_train, x_val, x_test = x[train_idxs], x[val_idxs], x[test_idxs]
        y_train, y_val, y_test = y[train_idxs], y[val_idxs], y[test_idxs]

        train_dataset = TimeSeriesDataset(x_train, y_train, seq_len=seq_len, pred_len=pred_len)
        val_dataset = TimeSeriesDataset(x_val, y_val, seq_len=seq_len, pred_len=pred_len)
        test_dataset = TimeSeriesDataset(x_test, y_test, seq_len=seq_len, pred_len=pred_len)
        cross_val_dataset = TimeSeriesDataset(x_cross_val, y_cross_val, seq_len=seq_len, pred_len=pred_len)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        cross_val_loader = DataLoader(cross_val_dataset, batch_size=64, shuffle=False)

        train_loss, test_loss, val_loss = train(model, train_loader, val_loader, test_loader, epochs=epochs,
                                                lr=lr, loss_fn=loss_fn, es_p=es_p, es_d=es_d)
        cross_val_loss = test_model(model, cross_val_loader, loss_fn=loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        cross_val_losses.append(cross_val_loss)
        print(f"Cross validation loss: {cross_val_loss:.8f}")
    return train_losses, val_losses, test_losses, cross_val_losses


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


def predict_ts_model(model: nn.Module, x: np.ndarray, y: np.ndarray, seq_len: int, pred_len: int):
    model.to(device)
    model.eval()
    dataset = TimeSeriesDataset(x, y, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predictions = np.zeros((0, seq_len), dtype=np.float32)
    true = np.zeros((0, seq_len), dtype=np.float32)
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            preds = model(features)
            preds = preds.cpu().numpy()
            labels = labels.numpy()
            predictions = np.vstack((predictions, preds))
            true = np.vstack((true, labels))
    return predictions, true


def print_model_evaluation(model: nn.Module, x: np.ndarray, y: np.ndarray,
                           x_unnorm: np.ndarray, seq_len: int, pred_len: int):
    preds, true = predict_ts_model(model, x, y, seq_len=seq_len, pred_len=pred_len)

    def std_denormalize(arr: np.ndarray, original_arr: np.ndarray, axis=0):
        return arr * original_arr.std(axis=axis, keepdims=True) + original_arr.mean(axis=axis, keepdims=True)

    preds = std_denormalize(preds, x_unnorm[:, 0])
    true = std_denormalize(true, x_unnorm[:, 0])

    print_evaluation_info(preds, true)


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

    plot_predictions(preds[-1000:], true[-1000:])


def plot_losses(train_losses: list, val_losses: list, test_losses: list):
    fig, axs = plt.subplots(nrows=1, ncols=len(train_losses), figsize=(8 * len(train_losses), 7))
    for i in range(len(train_losses)):
        axs[i].set_title(f"{i+1} fold")
        axs[i].plot(train_losses[i], label="train_loss", color="g")
        axs[i].plot(val_losses[i], label="val_loss", color="b")
        axs[i].plot(test_losses[i], label="test_loss", color="r")
        axs[i].legend()
    plt.show()


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
