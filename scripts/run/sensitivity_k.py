import os
import warnings
from pprint import pprint

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset, dataset_names

from numpy.linalg import LinAlgError

from src.workflows.sensitivity_on_k import SensAnalysisWorkflow

pprint(dataset_names)

warnings.simplefilter('ignore', UserWarning)

DS = 'nn5_daily_without_missing'
# DS = 'solar-energy'
# DS = 'traffic_nips'
# DS = 'electricity_nips'
# DS = 'm4_daily'
# DS = 'taxi_30min'
# DS = 'm4_hourly'
# DS = 'm4_weekly'

dataset = get_dataset(DS, regenerate=False)

train = list(dataset.train)
train = [x['target'] for x in train]
print(len(train))

for i, ds in enumerate(train):
    print(i)
    file_name = f'{DS}_ts_{i}.csv'
    if file_name in os.listdir('results/sensitivity_k'):
        continue

    series = pd.Series(ds)
    #
    try:
        # results = sensitivity_analysis(series.head(300), k=K, h=H)
        results = SensAnalysisWorkflow.on_k(series)
    except (ValueError, LinAlgError, IndexError) as e:
        continue
    #
    #
    results.to_csv(f'results/sensitivity_k/{file_name}')
