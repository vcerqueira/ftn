import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from src.preprocessing.lags import series_as_supervised


class ForecastedTrajectoryNeighbors:
    """
    Forecasted Trajectory Neighbors
    Improving Multi-step Forecasts with Neighbors Adjustments
    """

    def __init__(self, n_neighbors: int):
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
        self.Y_tr = pd.DataFrame()

    def fit(self, Y: pd.DataFrame):
        """
        Y: Target variables in the training data
        A pd.DataFrame with shape (n_observations, forecasting_horizon)

        Fitting in this case means indexation of training data
        """
        self.Y_tr = Y.copy()
        self.knn.fit(Y, Y[Y.columns[0]])

    def predict(self, Y_hat: pd.DataFrame, return_knn: bool = False):
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

        if return_knn:
            return knn_hat, k_neighbors

        return knn_hat


class FTNSmoothOperator(ForecastedTrajectoryNeighbors):
    """
    FTN with a smoooooth operator
    """

    def __init__(self, n_neighbors: int):
        super().__init__(n_neighbors)

    def fit(self, Y: pd.DataFrame):
        horizon = Y.shape[1]

        y_unrolled = Y.iloc[:, 0]

        y_smoothed = y_unrolled.ewm(alpha=.6).mean()
        y_smoothed.name = 'Series'

        _, Y_sm = series_as_supervised(series=y_smoothed, n_lags=0, horizon=horizon)

        self.Y_tr = Y_sm.copy()
        self.knn.fit(self.Y_tr, self.Y_tr[self.Y_tr.columns[0]])


class FTNAlpha(ForecastedTrajectoryNeighbors):
    """
    FTN with alpha
    """

    def __init__(self, n_neighbors: int, alpha: float):
        super().__init__(n_neighbors)

        self.alpha = alpha

    def predict(self, Y_hat: pd.DataFrame, return_knn: bool = False):
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

        preds = (knn_hat.values * self.alpha) + (Y_hat.values * (1 - self.alpha))
        preds = pd.DataFrame(preds)

        if return_knn:
            return preds, k_neighbors

        return preds
