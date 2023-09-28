import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from src.evaluation.montecarlo import MonteCarloCV
from src.workflows.lag_optimization import optimize_lag_size
from src.workflows.lgbm_params_optim import optimize_params
from src.workflows.testing_fourier_features import test_fourier_terms
from src.preprocessing.lags import series_as_supervised

from src.evaluation.metrics import ErrorOverHorizon
from src.methods.dirrec import DirRec
from src.methods.direct import Direct
from src.methods.recursive import Recursive
from src.methods.ftn import ForecastedTrajectoryNeighbors, FTNSmoothOperator, FTNAlpha
from src.methods.multi_ensemble import MultiOutputHeterogeneousEnsemble
from src.regression.classical import LazyForecastsR, ModelForecastR

from config import (CV_SPLIT_TRAIN_SIZE,
                    CV_SPLIT_TEST_SIZE,
                    CV_N_SPLITS,
                    HORIZON,
                    N_LAGS,
                    MODELS,
                    METHOD,
                    FTN_K,
                    APPLY_DIFF)

CLASSICAL_FORECASTERS = ['ARIMA', 'SNAIVE', 'THETA']


class MainWorkflow:

    @classmethod
    def cross_validation(cls, series: pd.Series, frequency: int):
        """
        :param series: univariate time series as pd.Series
        :param frequency: time series frequency
        """

        assert isinstance(series, pd.Series)

        cv = MonteCarloCV(n_splits=CV_N_SPLITS,
                          train_size=CV_SPLIT_TRAIN_SIZE,
                          test_size=CV_SPLIT_TEST_SIZE,
                          gap=0)

        splits = cv.split(series)

        err_list, times_list = [], []
        for i, (tr_idx, ts_idx) in enumerate(splits):
            print(f'CV split: {i}')

            y_train = series[series.index[tr_idx[0]]:series.index[tr_idx[-1]]]
            y_test = series[series.index[ts_idx[0]]:series.index[ts_idx[-1]]]

            err, times, predictions, Y_ts_or = cls.cv_iter(y_train=y_train,
                                                           y_test=y_test,
                                                           freq=frequency)

            err_list.append(err)
            times_list.append(times)

        return err_list, times_list, predictions, Y_ts_or

    @staticmethod
    def get_input_output_pairs(y_train: pd.Series, y_test: pd.Series, params: Dict):
        y_test_or = y_test.copy()

        if APPLY_DIFF:
            y_test = pd.concat([y_train.tail(1), y_test])
            y_train = y_train.diff()[1:]
            y_test = y_test.diff()[1:]

        try:
            n_lags = optimize_lag_size(y_train, params, N_LAGS)
        except ValueError:
            n_lags = 3

        data_params = {'n_lags': n_lags, 'horizon': HORIZON, }

        X_tr, Y_tr = series_as_supervised(y_train, **data_params)
        X_ts, Y_ts = series_as_supervised(y_test, **data_params)
        X_ts_or, Y_ts_or = series_as_supervised(y_test_or, **data_params)

        return X_tr, Y_tr, X_ts, Y_ts, X_ts_or, Y_ts_or, n_lags, y_train, y_test, y_test_or

    @staticmethod
    def get_methods(frequency: int, params: Dict):
        MSF_METHODS = {
            'DirRec': DirRec(MODELS[METHOD](params=params)),
            'Recursive': Recursive(MODELS[METHOD](params=params)),
            'Direct': Direct(MODELS[METHOD](params=params)),
            'KNN': KNeighborsRegressor(n_neighbors=FTN_K),
            'Ensemble': MultiOutputHeterogeneousEnsemble(),
            'ARIMA': ModelForecastR(frequency=frequency, method='arima101'),
            'SNAIVE': LazyForecastsR(frequency=frequency, method='snaive'),
            'THETA': LazyForecastsR(frequency=1, method='theta'),
        }

        return MSF_METHODS

    @classmethod
    def cv_iter(cls,
                y_train: pd.Series,
                y_test: pd.Series,
                freq: int):
        """
        Training and testing cycle within CV

        :param y_train: training series
        :param y_test: testing/validation series
        :param frequency: frequency of ts

        :return: tuple (error, execution time)
        """

        params = optimize_params(y_train)

        X_tr, Y_tr, X_ts, Y_ts, X_ts_or, Y_ts_or, n_lags, y_train, y_test, y_test_or = \
            cls.get_input_output_pairs(y_train, y_test, params)

        X_tr, X_ts = test_fourier_terms(X_tr, Y_tr, X_ts, freq, params)

        models = cls.get_methods(freq, params)

        methods_time = {}
        base_predictions = {}
        for model in models:
            print(f'Forecasting with method: {model}')
            start_time = time.time()

            if model in CLASSICAL_FORECASTERS:
                models[model].fit(y_train)
                preds = models[model].ms_predict(y_train, y_test, h=HORIZON, k=n_lags)

                preds = pd.DataFrame(preds.values, columns=Y_tr.columns)
                preds = preds.tail(Y_ts.shape[0])
                preds.index = Y_ts.index

                base_predictions[model] = preds
                methods_time[model] = time.time() - start_time
            else:
                models[model].fit(X_tr, Y_tr)
                preds = models[model].predict(X_ts)
                if isinstance(preds, np.ndarray):
                    preds = pd.DataFrame(preds, columns=Y_tr.columns)

                preds.index = Y_ts.index

                base_predictions[model] = preds
                methods_time[model] = time.time() - start_time

                if model == 'Ensemble':
                    preds = models[model].predict(X_ts, use_ftn=True)
                    if isinstance(preds, np.ndarray):
                        preds = pd.DataFrame(preds, columns=Y_tr.columns)

                    preds.index = Y_ts.index

                    base_predictions['BaseFTNEnsemble'] = preds
                    methods_time['BaseFTNEnsemble'] = time.time() - start_time

        print('Fitting FTN')
        ftn = ForecastedTrajectoryNeighbors(n_neighbors=FTN_K)
        ftn.fit(Y_tr)

        print('Fitting FTN(Smooth)')
        ftn_sm = FTNSmoothOperator(n_neighbors=FTN_K)
        ftn_sm.fit(Y_tr)

        print('Fitting FTN(alpha)')
        ftn_alpha = FTNAlpha(n_neighbors=FTN_K, alpha=0.5)
        ftn_alpha.fit(Y_tr)

        print('Forecasting')

        meta_predictions = {}
        print('---Base FTN')
        for pred_ in base_predictions:
            start_time = time.time()
            localize_pred = ftn.predict(base_predictions[pred_])
            localize_pred.index = base_predictions[pred_].index
            localize_pred.columns = base_predictions[pred_].columns

            meta_predictions[f'{pred_}+FTN'] = localize_pred
            end_time = time.time()
            methods_time[f'{pred_}+FTN'] = methods_time[pred_] + end_time - start_time

        meta_sm_predictions = {}
        print('---Smoothed FTN')
        for pred_ in base_predictions:
            start_time = time.time()
            localize_pred = ftn_sm.predict(base_predictions[pred_])
            localize_pred.index = base_predictions[pred_].index
            localize_pred.columns = base_predictions[pred_].columns

            meta_sm_predictions[f'{pred_}+FTN(Smooth)'] = localize_pred
            end_time = time.time()
            methods_time[f'{pred_}+FTN(Smooth)'] = methods_time[pred_] + end_time - start_time

        meta_alpha_predictions = {}
        print('---Alpha FTN')
        for pred_ in base_predictions:
            start_time = time.time()
            localize_pred = ftn_alpha.predict(base_predictions[pred_])
            localize_pred.index = base_predictions[pred_].index
            localize_pred.columns = base_predictions[pred_].columns

            meta_alpha_predictions[f'{pred_}+FTN(Alpha)'] = localize_pred
            end_time = time.time()
            methods_time[f'{pred_}+FTN(Alpha)'] = methods_time[pred_] + end_time - start_time

        predictions = {**base_predictions,
                       **meta_predictions,
                       **meta_sm_predictions,
                       **meta_alpha_predictions}

        if APPLY_DIFF:
            for mod in predictions:
                # mod = 'Direct+FTN'
                preds = predictions[mod]

                preds_or = preds.copy()
                for i, r in preds.iterrows():
                    # latest_val = y_test_or[r.name:].iloc[0]
                    latest_val = X_ts_or.loc[r.name, 'Series(t)']

                    diff_inv_vals = np.r_[latest_val, r.values].cumsum()[1:]

                    preds_or.loc[r.name] = diff_inv_vals

                predictions[mod] = preds_or

        err = ErrorOverHorizon.mean_squared_error(predictions, Y_ts_or)
        # err = ErrorOverHorizon.mean_absolute_error(predictions, Y_ts)

        return err, methods_time, predictions, Y_ts_or
