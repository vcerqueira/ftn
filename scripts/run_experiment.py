import os
import warnings
from pprint import pprint

import pandas as pd
from numpy.linalg import LinAlgError
from gluonts.dataset.repository.datasets import get_dataset, dataset_names

from workflows.cross_val import cross_val_workflow

warnings.simplefilter("ignore", UserWarning)

pprint(dataset_names)

ALL_DATASETS = [
    'nn5_daily_without_missing',
    'solar-energy',
    'traffic_nips',
    'electricity_nips',
    'm4_daily',
    'taxi_30min',
    'm4_hourly',
    'm4_weekly'
]

K = 5
H = 18

DS = 'nn5_daily_without_missing'

dataset = get_dataset(DS, regenerate=False)

train = list(dataset.train)
train = [x['target'] for x in train]

for i, ds in enumerate(train):
    print(i)
    file_name = f'{DS}_ts_{i}.csv'
    if file_name in os.listdir('results'):
        continue

    series = pd.Series(ds)

    try:
        series_result = cross_val_workflow(series, k=K, h=H)
    except (ValueError, LinAlgError, IndexError) as e:
        continue

    if len(series_result) == 0:
        continue

    err_df = pd.concat(series_result)
    err_avg = err_df.groupby(err_df.index).mean()

    err_avg.to_csv(f'results/{file_name}')
