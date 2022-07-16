import os

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

DATA_DIR = 'results/'

files = np.asarray(os.listdir(DATA_DIR))

err_data = []
for file in files:
    print(file)

    file_path = f'{DATA_DIR}{file}'
    try:
        df = pd.read_csv(file_path)
    except EmptyDataError:
        continue
    df.set_index('Unnamed: 0', inplace=True)
    h = df.index.str.replace('t\+', '').astype(int).to_series().values
    df = df.iloc[h.argsort(), :]
    avg_err = df.iloc[1:18, :].mean()

    err_data.append(avg_err)

err_df = pd.concat(err_data, axis=1).T

err_df.rank(axis=1).mean()
print(err_df.rank(axis=1).mean().sort_values())
