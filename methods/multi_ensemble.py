import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from regression.algorithms import METHODS, METHODS_PARAMETERS
from common.utils import expand_grid_all


class MultiOutputHeterogeneousEnsemble:

    def __init__(self):
        self.models = {}
        self.err = {}
        self.time = {}
        self.failed = []
        self.selected_methods = []
        self.col_names = []
        self.best_model = ''

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.col_names = Y.columns

        for learning_method in METHODS:
            print(f'Creating {learning_method}')
            if len(METHODS_PARAMETERS[learning_method]) > 0:
                gs_df = expand_grid_all(METHODS_PARAMETERS[learning_method])

                n_gs = len(gs_df[[*gs_df][0]])
                for i in range(n_gs):
                    print(f'Training {i} out of {n_gs}')

                    pars = {k: gs_df[k][i] for k in gs_df}
                    pars = {p: pars[p] for p in pars if pars[p] is not None}
                    print(pars)

                    model = METHODS[learning_method](**pars)
                    start = time.time()
                    model.fit(X, Y)
                    end_t = time.time() - start

                    self.models[f'{learning_method}_{i}'] = model
                    self.time[f'{learning_method}_{i}'] = end_t
            else:
                model = METHODS[learning_method]()
                start = time.time()
                model.fit(X, Y)

                end_t = time.time() - start

                self.models[f'{learning_method}_0'] = model
                self.time[f'{learning_method}_0'] = end_t

    def fit_and_trim(self, X, Y, select_percentile: float = .75):

        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1)

        self.fit(X_train, Y_train)

        Y_hat = self.predict_all(X_valid)

        for m in Y_hat:
            self.err[m] = mean_absolute_error(Y_valid, Y_hat[m])

        err_series = pd.Series(self.err)
        self.best_model = err_series.sort_values().index[0]
        self.selected_methods = err_series[err_series < err_series.quantile(select_percentile)].index.tolist()

        self.models = {k: self.models[k] for k in self.selected_methods}

    def predict_all(self, X: pd.DataFrame):

        preds_all = {}
        for method_ in self.models:
            predictions = self.models[method_].predict(X)
            preds_all[method_] = pd.DataFrame(predictions, columns=self.col_names)

        return preds_all

    def predict_mean(self, X: pd.DataFrame):
        preds_all = self.predict_all(X)

        preds_arr = np.asarray([*preds_all.values()])

        preds_mean = preds_arr.mean(axis=0)
        preds_mean = pd.DataFrame(preds_mean, columns=self.col_names)

        return preds_mean
