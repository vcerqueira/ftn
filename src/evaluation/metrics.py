from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error


class ErrorOverHorizon:

    @staticmethod
    def mean_squared_error(predictions: Dict, actual: pd.DataFrame):
        err = {}
        for model in predictions:
            err_k = {h: mean_absolute_error(
                actual[h],
                predictions[model][h]) for h in actual.columns}

            err[model] = err_k

        err_df = pd.DataFrame(err)

        return err_df
