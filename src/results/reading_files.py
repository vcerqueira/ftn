import os
import re
import warnings
from typing import List, Union, Optional

import pandas as pd

warnings.filterwarnings('ignore')


class ReadFiles:
    DATA_DIR = 'results/performance/'

    @classmethod
    def read_all_files(cls, horizon: Optional[Union[List[int], int]]):
        files = os.listdir(cls.DATA_DIR)
        # files = os.listdir('results/performance/')

        if '.DS_Store' in files:
            files.remove('.DS_Store')

        results_l = {}
        for file in files:
            # file = files[0]
            file_path = f'{cls.DATA_DIR}{file}'
            # file_path = f'{"results/performance/"}{file}'

            try:
                file_df = pd.read_csv(file_path, index_col='Unnamed: 0')
            except pd.errors.EmptyDataError:
                continue

            if file_df.shape[0] < 1:
                continue

            h_str = file_df.index.str.replace('Series\(t\+', '').str.replace('\)', '')
            h = h_str.astype(int).to_series().values

            file_df = file_df.iloc[h.argsort(), :]

            if horizon is None:
                average_loss = file_df.mean()
            else:
                average_loss = file_df.iloc[horizon, :].mean()

            results_l[file] = average_loss

        results_df = pd.concat(results_l, axis=1).T

        results_df.columns = [re.sub('SNAIVE', 'Naive', x) for x in results_df.columns]
        results_df.columns = [re.sub('THETA', 'Theta', x) for x in results_df.columns]
        results_df.columns = [re.sub('BaseFTNEnsemble', 'FTNE', x) for x in results_df.columns]

        return results_df
