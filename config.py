from sklearn.linear_model import RidgeCV, LassoCV
from src.regression.lgbm_random import LightGBMRegressorRS
from src.regression.lgbm_optuna import LGBMOptuna

CV_N_SPLITS = 5
CV_SPLIT_TRAIN_SIZE = 0.6
CV_SPLIT_TEST_SIZE = 0.2
FTN_K = 150
APPLY_DIFF = True
METHOD = 'LGBM_RS'

N_LAGS_DEFAULT = 10
N_LAGS = [2, 4, 6, 8, 10, 20, 30, 50]
HORIZON = 18

FREQUENCY = {
    'nn5_daily_without_missing': 7,
    'solar-energy': 24,
    'traffic_nips': 24,
    'electricity_nips': 24,
    'm4_daily': 7,
    'taxi_30min': 48,
    'm4_hourly': 24,
    'm4_weekly': 52
}

MODELS = {'LGBM_RS': LightGBMRegressorRS,
          'LGBM_OPT': LGBMOptuna,
          'LASSO': LassoCV,
          'RIDGE': RidgeCV}

MAIN_COLOR = '#2c5f78'
MAIN_COLOR2 = '#51acc5'
