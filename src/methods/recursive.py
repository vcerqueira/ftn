import pandas as pd
import numpy as np


class Recursive:

    def __init__(self, model):
        self.model = model
        self.fh = 0
        self.col_names = ''

    def fit(self, X, Y):
        self.fh = Y.shape[1]
        self.col_names = Y.columns
        self.model.fit(X, Y[Y.columns[0]])

    def predict(self, X):
        assert isinstance(X, pd.DataFrame), 'not pd.DF'

        Y_hat = pd.DataFrame(np.zeros((X.shape[0], self.fh)),
                             columns=self.col_names)

        yh = self.model.predict(X)
        Y_hat['Series(t+1)'] = yh

        X_ = X.copy()
        for i in range(2, self.fh + 1):
            X_.iloc[:, :-1] = X_.iloc[:, 1:].values
            X_['Series(t)'] = yh

            yh = self.model.predict(X_)

            Y_hat[f'Series(t+{i})'] = yh

        return Y_hat
