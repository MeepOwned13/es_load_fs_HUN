import numpy as np
import torch
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from math import sqrt as math_sqrt
import pandas as pd
from timeit import default_timer as timer
import models.trainer_lib as tl
import os
import argparse
import json
from configs import CONFIGS, load_data, make_rf_data
import joblib


# region Helper functions

def calc_metrics(p, t):
    loss = nn.MSELoss()(torch.tensor(p), torch.tensor(t)).item()
    mae = round(np.mean(np.abs(p - t)), 4)
    mse = round(loss, 4)
    rmse = round(math_sqrt(loss), 4)
    mape = round(tl.mape(p, t) * 100, 4)
    mpe = round(tl.mpe(p, t) * 100, 4)

    return mae, mse, rmse, mape, mpe


# endregion


# region Argument parsing and setup

parser = argparse.ArgumentParser(description='Evaluate final models and strategies')
parser.add_argument('-c', '--config', choices=list(CONFIGS.keys()) + ['mimo_rf'],
                    required=True, help='Model and strategy to evaluate')
parser.add_argument('-r', '--repeat', type=int, default=6)
parser.add_argument('-sw', '--skip_write', action='store_true',
                    help="Don't write results to file")
parser.add_argument('-sm', '--save_model', action='store_true',
                    default=False, help="Save best model for each fold")
args = parser.parse_args()
config = CONFIGS[args.config]
write_file = not args.skip_write
save_model = args.save_model

N_SPLITS = config['n_splits']
EPOCHS = config['epochs']
LR = config['lr']
BATCH_SIZE = config['batch_size']
ES_P = config['es_p']
FILE_NAME = config['file_name']

wrapper = config['wrapper'](
    config['model'](**config['model_params']),
    config['seq_len'], config['pred_len'],
    **config['extra_strat_params']
) if args.config != 'mimo_rf' else None

ts_cv = TimeSeriesSplit(n_splits=N_SPLITS)

X, y = load_data(config['load_modifier'])

# endregion

# check if file exists
if (write_file or save_model) and not os.path.exists(FILE_NAME):
    if not os.path.exists('final_eval_results'):
        os.mkdir('final_eval_results')
    df = pd.DataFrame(columns=['Fold', 'Hour', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MPE', 'Train Time', 'Pred Time'])
    df.to_csv(FILE_NAME, index=False)

# make file to save models to
if save_model and not os.path.exists(f'final_eval_results/{args.config}'):
    os.mkdir(f'final_eval_results/{args.config}')

# load previous best models, or create new dict
if save_model and os.path.exists('final_eval_results/strategies.json'):
    with open('final_eval_results/strategies.json', 'r') as f:
        strat_json = json.load(f)
else:
    strat_json = {k: {} for k in CONFIGS.keys()}

# add config if it previously didn't exist
if args.config not in strat_json:
    strat_json[args.config] = {}

full_run_start = timer()
for _ in range(args.repeat):
    for i, (train_idxs, test_idxs) in enumerate(ts_cv.split(X)):
        st_time = timer()
        print(f"[Fold {i + 1}]", end=" ", flush=True)

        train_val_sp: int = len(X) // (N_SPLITS + 1) // 8
        val_idxs = train_idxs[-train_val_sp:]
        train_idxs = train_idxs[:-train_val_sp]

        x_train, x_val, x_test = X[train_idxs], X[val_idxs], X[test_idxs]
        y_train, y_val, y_test = y[train_idxs], y[val_idxs], y[test_idxs]

        rf = None
        rf_test_x = None
        rf_test_y = None
        if args.config == 'mimo_rf':
            rf_X, rf_y = make_rf_data(x_train, y_train, config['seq_len'], config['pred_len'])

            rf = RandomForestRegressor(n_jobs=os.cpu_count()-2, **config['model_params'])
            rf.fit(rf_X, rf_y)

            rf_test_x, rf_test_y = make_rf_data(x_test, y_test, config['seq_len'], config['pred_len'])
        else:
            wrapper.init_strategy()
            _ = wrapper.train_strategy(x_train, y_train, x_val, y_val, x_test, y_test,
                                       epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, loss_fn=nn.MSELoss(),
                                       es_p=ES_P, es_d=0, verbose=0, cp=True)

        train_finish = timer()
        training_time = train_finish - st_time

        preds, true = wrapper.predict(x_test, y_test) if args.config != 'mimo_rf' else (rf.predict(rf_test_x), rf_test_y)

        pred_finish = timer()
        prediction_time = pred_finish - train_finish

        mae_all, mse_all, rmse_all, mape_all, mpe_all = calc_metrics(preds, true)

        # hour 0 will be the average of all hours
        df = pd.DataFrame({'Fold': i + 1, 'Hour': 0, 'MAE': mae_all, 'MSE': mse_all, 'RMSE': rmse_all,
                           'MAPE': mape_all, 'MPE': mpe_all, 'Train Time': training_time,
                           'Pred Time': prediction_time}, index=[0])

        for j in range(preds.shape[1]):
            mae, mse, rmse, mape, mpe = calc_metrics(preds[:, j], true[:, j])

            to_concat = pd.DataFrame({'Fold': i + 1, 'Hour': j + 1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                                      'MAPE': mape, 'MPE': mpe, 'Train Time': training_time,
                                      'Pred Time': prediction_time}, index=[j])

            df = pd.concat([df, to_concat], axis='rows', ignore_index=True)

        if write_file:
            df.to_csv(FILE_NAME, mode='a', header=False, index=False)

        # saving model if it's the best one for the fold
        if save_model:
            fold = f"fold_{i+1}"
            if fold in strat_json[args.config] and strat_json[args.config][fold]['RMSE'] < rmse_all:
                pass
            else:
                if args.config != 'mimo_rf':
                    wrapper.save_state(f'final_eval_results/{args.config}/{fold}.pt')
                else:
                    joblib.dump(rf, f'final_eval_results/{args.config}/{fold}.joblib', compress=9)
                strat_json[args.config][fold] = {
                    'RMSE': rmse_all,
                    'model': f'final_eval_results/{args.config}/{fold}.pt',
                }
                with open('final_eval_results/strategies.json', 'w') as f:
                    json.dump(strat_json, f, indent=4)

        print(f"- RMSE loss: {rmse_all:.3f} - "
              f"Train time: {training_time / 60:.2f} min - "
              f"Pred time: {prediction_time:.3f} sec")

    print("=" * 50, f"Elapsed time: {(timer() - full_run_start) / 60:.2f} mins")
