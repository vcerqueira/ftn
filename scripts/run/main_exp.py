import os
import warnings

import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset

from src.workflows.ftn_validation import MainWorkflow

from config import FREQUENCY

warnings.simplefilter('ignore', UserWarning)

PERF_DIR = 'results/performance'
TIME_DIR = 'results/execution_time'
MIN_SAMPLE_SIZE = 500

ALL_DATASETS = {
    'nn5_daily_without_missing',
    'solar-energy',
    'traffic_nips',
    'electricity_nips',
    'taxi_30min',
    'm4_hourly',
    'm4_weekly',
    'm4_daily'
}

DS = 'nn5_daily_without_missing'
freq = FREQUENCY[DS]

dataset = get_dataset(dataset_name=DS, regenerate=False)

train = list(dataset.train)
train_list = [pd.Series(ts['target'],
                        index=pd.date_range(start=ts['start'],
                                            freq=ts['start'].freq,
                                            periods=len(ts['target'])))
              for ts in train]

# train_list = [pd.Series(ds['target'], index=pd.date_range(start=ds['start'].to_timestamp(),
#                                                           freq=ds['start'].freq,
#                                                           periods=len(ds['target'])))
#               for ds in train]

print(len(train_list))
print(len(train_list[0]))

for i, series in enumerate(train_list):
    # i=0
    # series = train_list[i]

    if len(series) < MIN_SAMPLE_SIZE:
        continue

    print(f'{i}/{len(train_list)}')

    file_name = f'{DS}_{i}.csv'
    if file_name in os.listdir(PERF_DIR):
        continue
    else:
        pd.DataFrame().to_csv(f'{PERF_DIR}/{file_name}')

    # series = pd.Series(range(400), index=pd.date_range(start=pd.Timestamp('2020-01-01'), freq='D', periods=400))
    # series = pd.Series(np.random.random(400), index=pd.date_range(start=pd.Timestamp('2020-01-01'), freq='D', periods=400))
    series_result, time_results, predictions_, Y_ts_or_ = \
        MainWorkflow.cross_validation(series=series,
                                      frequency=freq)

    err_df = pd.concat(series_result)
    err_avg = err_df.groupby(err_df.index).mean()
    print(err_avg.mean().sort_values())

    time_avg = pd.DataFrame(time_results).mean()

    err_avg.to_csv(f'{PERF_DIR}/{file_name}')
    time_avg.to_csv(f'{TIME_DIR}/{file_name}')
