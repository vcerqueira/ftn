import pandas as pd

from src.preprocessing.lags import series_as_supervised
from config import LGBMOptuna, LightGBMRegressorRS


# def optimize_params(y_train: pd.Series):
#     X_tr, Y_tr = series_as_supervised(y_train, n_lags=5, horizon=1)
#
#     mod = LGBMOptuna(n_trials=200)
#
#     mod.fit(X_tr, Y_tr)
#
#     return mod.params

def optimize_params(y_train: pd.Series):
    X_tr, Y_tr = series_as_supervised(y_train, n_lags=5, horizon=1)

    mod = LightGBMRegressorRS(iters=200)

    mod.fit(X_tr, Y_tr)

    return mod.model.best_params_
