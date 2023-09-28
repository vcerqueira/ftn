import copy
from typing import Optional

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri

from src.regression.r_objects \
    import (tbats_model_fit,
            tbats_predict,
            tbats_update,
            ets_model_fit,
            ets_predict,
            ets_update,
            auto_arima_model_fit,
            auto_arima_predict,
            arima101_model_fit,
            auto_arima_update,
            thetaf_forecast,
            snaive_forecast,
            model_forecast)


class ModelForecastR:

    def __init__(self, method: str, frequency: Optional[int]):

        assert method in ['arima', 'arima101', 'ets', 'tbats']

        if frequency is None:
            self.frequency = 1
        else:
            self.frequency = frequency

        self.method = method
        self.model = None
        self.model_is_fit = False

    def fit(self, y: np.ndarray):
        pandas2ri.activate()

        y_fit = copy.deepcopy(y)

        if isinstance(y, np.ndarray):
            y_fit = pd.Series(y_fit)

        y_r = pandas2ri.py2rpy_pandasseries(y_fit)

        if self.method == 'arima':
            fit_fun = auto_arima_model_fit
        elif self.method == 'ets':
            fit_fun = ets_model_fit
        elif self.method == 'arima101':
            fit_fun = arima101_model_fit
        else:
            fit_fun = tbats_model_fit

        self.model = fit_fun(y_r, self.frequency)
        self.model_is_fit = True
        pandas2ri.deactivate()

        return self

    def predict(self, y):
        assert self.model_is_fit

        pandas2ri.activate()

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        y_r = pandas2ri.py2rpy_pandasseries(y)

        if self.method == 'arima':
            predict_fun = auto_arima_predict
        elif self.method == 'ets':
            predict_fun = ets_predict
        elif self.method == 'arima101':
            predict_fun = auto_arima_predict
        else:
            predict_fun = tbats_predict

        y_hat = predict_fun(self.model, y_r)
        y_hat = np.asarray(y_hat)

        pandas2ri.deactivate()

        return y_hat

    def update_model(self, y):
        assert self.model_is_fit

        pandas2ri.activate()

        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        y_r = pandas2ri.py2rpy_pandasseries(y)

        if self.method == 'arima':
            update_fun = auto_arima_update
        elif self.method == 'ets':
            update_fun = ets_update
        elif self.method == 'arima101':
            update_fun = auto_arima_update
        else:
            update_fun = tbats_update

        self.model = update_fun(self.model, y_r)

        pandas2ri.deactivate()

        return self

    def forecast(self, h: int):
        assert self.model_is_fit

        pandas2ri.activate()

        y_hat = model_forecast(self.model, h)
        y_hat = np.asarray(y_hat)

        pandas2ri.deactivate()

        return y_hat

    def ms_predict(self, y_train, y_test, h, k):
        self.fit(y_train)

        # y_complete = np.concatenate([y_train, y_test])
        y_complete = pd.concat([y_train, y_test])
        y_complete = y_complete.reset_index(drop=True)

        predictions = []
        for i in range(len(y_train) - k - h, len(y_complete) - h + 1):
            y_iter = y_complete[:i]

            self.update_model(y=y_iter)
            yh = self.forecast(h=h)
            predictions.append(yh)

        predictions = np.asarray(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.columns = \
            [f'Series(t+{i})' + str(i)
             for i in range(1, predictions.shape[1] + 1)]

        predictions.reset_index(drop=True, inplace=True)

        return predictions

    def ms_predict2(self, y_train, y_test, h, k):
        self.fit(y_train)

        y_complete = np.concatenate([y_train, y_test])

        yh_ts = []
        for i in range(len(y_train) - k + 1, len(y_complete) - h + 1):
            y_iter = y_complete[:i]

            self.update_model(y=y_iter)
            yh = self.forecast(h=h)
            yh_ts.append(yh)

        yh_tr = []
        for i in range(k, len(y_train) - h + 1):
            if i < h + 1:
                continue

            y_iter = y_train[:i]

            self.update_model(y=y_iter)
            yh = self.forecast(h=h)
            yh_tr.append(yh)

        yh_tr = np.asarray(yh_tr)
        yh_tr = pd.DataFrame(yh_tr)
        yh_tr.columns = \
            [f'Series(t+{i})' + str(i)
             for i in range(1, yh_tr.shape[1] + 1)]

        yh_tr.reset_index(drop=True, inplace=True)

        #

        yh_ts = np.asarray(yh_ts)
        yh_ts = pd.DataFrame(yh_ts)
        yh_ts.columns = \
            [f'Series(t+{i})'
             for i in range(1, yh_ts.shape[1] + 1)]

        yh_ts.reset_index(drop=True, inplace=True)

        # yh_tr = predictions[:-len(y_test)]
        # yh_ts = predictions[-len(y_test):]

        return yh_tr, yh_ts


class LazyForecastsR:
    N_BURN_IN = 100

    def __init__(self, method: str, frequency: Optional[int]):
        assert method in ['snaive', 'theta']

        if frequency is None:
            self.frequency = 1
        else:
            self.frequency = frequency

        self.method = method

    def forecast(self, y_train: np.ndarray, h: int):
        pandas2ri.activate()

        if self.method == 'snaive':
            forecast_fun = snaive_forecast
        else:
            forecast_fun = thetaf_forecast

        y_hat = forecast_fun(y_train, self.frequency, h)
        y_hat = np.asarray(y_hat)

        pandas2ri.deactivate()

        return y_hat

    def fit(self, y: np.ndarray):
        pass

    def ms_predict(self, y_train, y_test, h, k):
        y_complete = np.concatenate([y_train, y_test])

        predictions = []
        for i in range(self.N_BURN_IN, len(y_complete) - h + 1):
            y_iter = y_complete[:i]

            yh = self.forecast(y_iter, h)
            predictions.append(yh)

        predictions = np.asarray(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.columns = \
            [f'Series(t+{i})' + str(i)
             for i in range(1, predictions.shape[1] + 1)]

        predictions.reset_index(drop=True, inplace=True)

        # yh_tr = predictions[:-len(y_test)]
        # predictions = predictions[-len(y_test):]

        return predictions

    def ms_predict2(self, y_train, y_test, h, k):
        y_complete = np.concatenate([y_train, y_test])

        yh_ts = []
        for i in range(len(y_train) - k + 1, len(y_complete) - h + 1):
            if i < h + 1:
                continue

            y_iter = y_complete[:i]

            yh = self.forecast(y_iter, h)
            yh_ts.append(yh)

        yh_tr = []
        for i in range(k, len(y_train) - h + 1):
            if i < h + 1:
                continue

            y_iter = y_train[:i]

            yh = self.forecast(y_iter, h)
            yh_tr.append(yh)

        yh_tr = np.asarray(yh_tr)
        yh_tr = pd.DataFrame(yh_tr)
        yh_tr.columns = \
            [f'Series(t+{i})' + str(i)
             for i in range(1, yh_tr.shape[1] + 1)]

        yh_tr.reset_index(drop=True, inplace=True)

        #

        yh_ts = np.asarray(yh_ts)
        yh_ts = pd.DataFrame(yh_ts)
        yh_ts.columns = \
            ['t+' + str(i)
             for i in range(1, yh_ts.shape[1] + 1)]

        yh_ts.reset_index(drop=True, inplace=True)

        # yh_tr = predictions[:-len(y_test)]
        # yh_ts = predictions[-len(y_test):]

        return yh_tr, yh_ts


"""arima = ModelForecastR(method='arima', frequency=None)
ets = ModelForecastR(method='ets', frequency=None)
tbats = ModelForecastR(method='tbats', frequency=None)
snaive = LazyForecastsR(method='snaive', frequency=None)
theta = LazyForecastsR(method='theta', frequency=None)

from sklearn.model_selection import train_test_split

y_train, y_test = train_test_split(series, test_size=0.3)
theta.fit(y_train)
y_hat = snaive.ms_predict2(y_train=y_train, y_test=y_test, h=18, k=2)
"""
