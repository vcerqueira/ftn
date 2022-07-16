import pandas as pd

from common.monte_carlo import MonteCarloCV
from common.tde import UnivariateTDE
from workflows.train_test_cycle import cval_cycle
from workflows.config import (CV_SPLIT_TRAIN_SIZE,
                              CV_SPLIT_TEST_SIZE,
                              CV_N_SPLITS,
                              APPLY_DIFF)


def cross_val_workflow(series, k, h):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if APPLY_DIFF:
        series = series.diff()

    df = UnivariateTDE(series, k=k, horizon=h)

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]

    mc = MonteCarloCV(n_splits=CV_N_SPLITS,
                      train_size=CV_SPLIT_TRAIN_SIZE,
                      test_size=CV_SPLIT_TEST_SIZE,
                      gap=h + k)

    err_list = []
    for tr_idx, ts_idx in mc.split(X, Y):
        X_tr = X.iloc[tr_idx, :]
        Y_tr = Y.iloc[tr_idx, :]

        X_ts = X.iloc[ts_idx, :]
        Y_ts = Y.iloc[ts_idx, :]

        print('Running inner pipeline')
        err = cval_cycle(X_tr, Y_tr, X_ts, Y_ts)
        err_list.append(err)

    return err_list
