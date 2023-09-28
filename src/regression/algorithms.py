from sklearn.ensemble \
    import (RandomForestRegressor,
            ExtraTreesRegressor,
            BaggingRegressor)
from sklearn.linear_model \
    import (Lasso,
            Ridge,
            OrthogonalMatchingPursuit,
            ElasticNet)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

METHODS = \
    dict(
        RandomForestRegressor=RandomForestRegressor,
        PLSRegression=PLSRegression,
        PLSCanonical=PLSCanonical,
        ExtraTreesRegressor=ExtraTreesRegressor,
        OrthogonalMatchingPursuit=OrthogonalMatchingPursuit,
        Lasso=Lasso,
        KNeighborsRegressor=KNeighborsRegressor,
        Ridge=Ridge,
        ElasticNet=ElasticNet,
        BaggingRegressor=BaggingRegressor,
    )

METHODS_PARAMETERS = \
    dict(
        RandomForestRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        ExtraTreesRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        OrthogonalMatchingPursuit={},
        Lasso={
            'alpha': [1, .5, .25, .75]
        },
        KNeighborsRegressor={
            'n_neighbors': [5, 10, 20],
            'weights': ['uniform', 'distance'],
        },
        Ridge={
            'alpha': [1, .5, .25, .75]
        },
        ElasticNet={
        },
        PLSRegression={
            'n_components': [2, 3]
        },
        PLSCanonical={
            'n_components': [2, 3]
        },
        BaggingRegressor={
            'n_estimators': [50, 100]
        },
    )
