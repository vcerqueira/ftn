import warnings

import pandas as pd

from src.evaluation.montecarlo import MonteCarloCV
from src.evaluation.metrics import ErrorOverHorizon
from src.methods.direct import Direct
from src.methods.ftn import FTNAlpha
from src.regression.lgbm_random import LightGBMRegressorRS
from src.workflows.ftn_validation import MainWorkflow
from src.workflows.lgbm_params_optim import optimize_params

from config import APPLY_DIFF

warnings.simplefilter('ignore', UserWarning)

K_OPTIONS = [1, 3, 5, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000, 2000]


class SensAnalysisWorkflow(MainWorkflow):

    @classmethod
    def on_k(cls, series):
        if APPLY_DIFF:
            series = series.diff()

        mc = MonteCarloCV(n_splits=1, train_size=.7, test_size=.25, gap=0)
        tr_idx, ts_idx = mc.split(series).__next__()

        y_train = pd.Series(series.values[tr_idx])
        y_test = pd.Series(series.values[ts_idx])

        params = optimize_params(y_train)

        X_tr, Y_tr, X_ts, Y_ts, X_ts_or, Y_ts_or, n_lags, y_train, y_test, y_test_or = cls.get_input_output_pairs(
            y_train, y_test, params)

        model = Direct(LightGBMRegressorRS())
        model.fit(X_tr, Y_tr)
        y_hat = model.predict(X_ts)
        y_hat = pd.DataFrame(y_hat, columns=Y_tr.columns)

        ftn_predictions = {}
        for k in K_OPTIONS:
            name_ = f'Recursive+FTN({k})'
            if k > Y_tr.shape[0]:
                k = Y_tr.shape[0]

            ftn = FTN(n_neighbors=k)
            ftn.fit(Y_tr)
            ftn_pred = ftn.predict(y_hat)
            ftn_predictions[name_] = ftn_pred

        ftn_predictions['Recursive'] = y_hat

        err_df = ErrorOverHorizon.mean_squared_error(ftn_predictions, Y_ts)

        return err_df

    @classmethod
    def on_data_size(cls, series):
        if APPLY_DIFF:
            series = series.diff()

        mc = MonteCarloCV(n_splits=1, train_size=.7, test_size=.25, gap=0)
        tr_idx, ts_idx = mc.split(series).__next__()

        y_train = pd.Series(series.values[tr_idx])
        y_test = pd.Series(series.values[ts_idx])

        params = optimize_params(y_train)

        X_tr, Y_tr, X_ts, Y_ts, X_ts_or, Y_ts_or, n_lags, y_train, y_test, y_test_or = cls.get_input_output_pairs(
            y_train, y_test, params)

        model = Direct(LightGBMRegressorRS())
        model.fit(X_tr, Y_tr)
        y_hat = model.predict(X_ts)
        y_hat = pd.DataFrame(y_hat, columns=Y_tr.columns)

        ftn = FTNAlpha(n_neighbors=100, alpha=0.5)
        ftn.fit(Y_tr)
        ftn_pred = ftn.predict(y_hat)
        ftn_pred.columns = y_hat.columns

        preds = {
            'base': y_hat,
            'ftn': ftn_pred,
        }

        err = ErrorOverHorizon.mean_squared_error(preds, Y_ts)

        err_pd = 100 * ((err['ftn'] - err['base']) / err['base'])

        return err_pd
