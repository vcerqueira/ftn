import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

from src.preprocessing.fourier import FourierTerms

from config import MODELS, METHOD


def test_fourier_terms(X_tr, Y_tr, X_ts, frequency, params):
    fourier = FourierTerms(n_terms=2, period=frequency)

    fourier_features = fourier.transform(X_tr.index)

    X_tr_ext = pd.concat([X_tr, fourier_features], axis=1)

    X_dev1, X_val1, y_dev, y_val = train_test_split(X_tr, Y_tr.iloc[:, 0], test_size=0.2, shuffle=False)
    X_dev2, X_val2, _, _ = train_test_split(X_tr_ext, Y_tr.iloc[:, 0], test_size=0.2, shuffle=False)

    mod1 = MODELS[METHOD](params=params)
    mod2 = MODELS[METHOD](params=params)

    mod1.fit(X_dev1, y_dev)
    mod2.fit(X_dev2, y_dev)

    pred1 = mod1.predict(X_val1)
    pred2 = mod2.predict(X_val2)

    err_mod1 = mae(pred1, y_val)
    err_mod2 = mae(pred2, y_val)

    if err_mod2 < err_mod1:
        # fourier helps
        test_features = fourier.transform(X_ts.index)
        X_ts_ext = pd.concat([X_ts, test_features], axis=1)

        return X_tr_ext, X_ts_ext
    else:
        return X_tr, X_ts
