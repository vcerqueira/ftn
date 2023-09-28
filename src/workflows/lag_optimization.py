from typing import List, Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split

from src.preprocessing.lags import series_as_supervised

from config import METHOD, MODELS

HORIZON = 1  # OPTIMIZING FOR ONE STEP AHEAD FORECASTING


def optimize_lag_size(y_train: pd.Series,
                      model_params: Dict,
                      n_lags_list: List):
    """
    Optimizing n_lags on one step ahead forecasting

    :param y_train: TRAINING SERIES AS PD.SERIES
    :param model_params: dict w model params
    :param n_lags_list: LIST OF LAGS TO TRY

    :return: BEST LAG
    """
    y_dev, y_val = train_test_split(y_train, test_size=0.2, shuffle=False)

    n_lags_err = {}
    for n_lags_ in n_lags_list:
        X_tr, Y_tr = series_as_supervised(y_dev, n_lags=n_lags_, horizon=HORIZON, return_Xy=True)
        X_ts, Y_ts = series_as_supervised(y_val, n_lags=n_lags_, horizon=HORIZON, return_Xy=True)

        mod = MODELS[METHOD](params=model_params)

        mod.fit(X_tr, Y_tr)
        preds = mod.predict(X_ts)

        n_lags_err[n_lags_] = mae(Y_ts, preds)

    best_ind = pd.Series(n_lags_err).argmin()

    best_n_lags = n_lags_list[best_ind]

    return best_n_lags
