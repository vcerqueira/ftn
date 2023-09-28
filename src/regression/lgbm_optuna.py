from typing import Optional, Dict

import optuna
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

VALIDATION_SIZE = 0.2


def objective_r(trial, X, y):
    train_x, valid_x, train_y, valid_y = \
        train_test_split(X, y, test_size=VALIDATION_SIZE, shuffle=True)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        'objective': 'regression',
        'metric': 'mean_absolute_error',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'linear_tree': True,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)

    err = mean_absolute_error(valid_y, preds)

    return err


def optimize_params(X, y, n_trials: int):
    func = lambda trial: objective_r(trial, X, y)

    study = optuna.create_study(direction='minimize')
    study.optimize(func, n_trials=n_trials)

    trial = study.best_trial

    return trial.params


class LGBMOptuna(BaseEstimator, RegressorMixin):

    def __init__(self, params: Optional[Dict] = None, n_trials: int = 100):
        self.model = None
        self.n_trials = n_trials
        self.params = params

    def fit(self, X, y):
        if self.params is None:
            self.params = optimize_params(X, y, n_trials=self.n_trials)

        dtrain = lgb.Dataset(X, label=y)

        self.model = lgb.train(self.params, dtrain)

    def predict(self, X):
        return self.model.predict(X)
