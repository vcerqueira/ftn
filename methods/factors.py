import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from methods.direct import Direct


class DynamicFactors:

    def __init__(self, model, min_var_expl: float = 0.8):
        self.min_var_expl = min_var_expl
        self.pca = PCA()
        self.scaler = StandardScaler()
        self.model = Direct(model)

    def fit(self, X, Y):
        Y_sc = self.scaler.fit_transform(Y)

        self.pca.fit(Y_sc)

        expl_var = self.pca.explained_variance_ratio_
        cs_exp_variance = np.cumsum(expl_var)

        n_components = np.where(cs_exp_variance >= self.min_var_expl)[0][0] + 1

        self.pca = PCA(n_components=n_components)
        self.pca.fit(Y_sc)

        Y_tr_t = self.pca.transform(self.scaler.transform(Y))

        self.model.fit(X, Y_tr_t)

    def predict(self, X):
        Y_hat_t = self.model.predict(X)
        Y_hat = self.pca.inverse_transform(Y_hat_t)
        Y_hat = self.scaler.inverse_transform(Y_hat)

        return Y_hat
