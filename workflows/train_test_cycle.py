import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

import warnings

warnings.simplefilter('ignore', UserWarning)

from common.error import multistep_mae
from methods.dirrec import DirRec
from methods.direct import Direct
from methods.factors import DynamicFactors
from methods.recursive import Recursive
from methods.ftn import ForecastedTrajectoryNeighbors
from methods.multi_ensemble import MultiOutputHeterogeneousEnsemble

from regression.lgbm import LightGBMRegressor

FTN_K = 20


def cval_cycle(X_tr, Y_tr, X_ts, Y_ts):
    """
    :param X_tr:
    :param Y_tr:
    :param X_ts:
    :param Y_ts:
    :return:
    """

    msf_methods = {
        'DFML': DynamicFactors(LightGBMRegressor()),
        'DirRec': DirRec(LightGBMRegressor()),
        'Recursive': Recursive(LightGBMRegressor()),
        'Direct': Direct(LightGBMRegressor()),
        'Lazy': KNeighborsRegressor(n_neighbors=10),
    }

    base_msf_predictions = {}
    for method_ in msf_methods:
        print(f'Training {method_}')
        msf_methods[method_].fit(X_tr, Y_tr)
        method_pred = msf_methods[method_].predict(X_ts)
        method_pred = pd.DataFrame(method_pred, columns=Y_tr.columns)

        base_msf_predictions[method_] = method_pred

    print('Training Ensemble')
    msf_methods['Ensemble'] = MultiOutputHeterogeneousEnsemble()
    msf_methods['Ensemble'].fit_and_trim(X_tr, Y_tr)

    ensemble_mo_mean = msf_methods['Ensemble'].predict_mean(X_ts)
    ensemble_mo_mean = pd.DataFrame(ensemble_mo_mean, columns=Y_tr.columns)
    base_msf_predictions['Ensemble'] = ensemble_mo_mean

    print('Fitting FTN')
    ftn = ForecastedTrajectoryNeighbors(n_neighbors=FTN_K)
    ftn.fit(Y_tr)

    localized_msf_predictions = {}
    for pred_ in base_msf_predictions:
        loc_pred = ftn.predict(base_msf_predictions[pred_])

        localized_msf_predictions[f'{pred_}_FTN'] = loc_pred

    predictions = {**base_msf_predictions, **localized_msf_predictions}

    err = multistep_mae(predictions, Y_ts)

    return err
