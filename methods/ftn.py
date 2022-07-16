import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


class ForecastedTrajectoryNeighbors:
    """
    FTN class object
    """

    def __init__(self, n_neighbors: int, weights='uniform'):
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        self.Y_tr = pd.DataFrame()

    def fit(self, Y: pd.DataFrame):
        """
        Y: Target variables in the training data
        A pd.DataFrame with shape (n_observations, forecasting_horizon)

        Fitting means simply indexation of training data
        """
        self.Y_tr = Y.copy()
        self.knn.fit(Y, Y[Y.columns[0]])

    def predict(self, Y_hat: pd.DataFrame):
        """
        Making predictions

        Y_hat: pd.DF with the multi-step forecasts of a base-model
        Y_hat has shape (n_test_observations, forecasting_horizon)
        """
        # getting nearest neighbors
        _, k_neighbors = self.knn.kneighbors(Y_hat)

        # averaging the targets of neighbors
        average_knn = [self.Y_tr.iloc[ind, :].mean() for ind in k_neighbors]

        # binding predictions in a dataframe
        knn_hat = pd.concat(average_knn, axis=1).T

        return knn_hat
