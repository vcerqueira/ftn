import warnings

import pandas as pd
from plotnine import *
from gluonts.dataset.repository.datasets import get_dataset

from src.results.reading_files import ReadFiles
from src.preprocessing.log import LogTransformation
from config import MAIN_COLOR, MAIN_COLOR2

warnings.simplefilter('ignore', UserWarning)

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

METHODS = ['Recursive', 'Direct', 'DirRec']

results = ReadFiles.read_all_files(horizon=None)

ds_size = {}
for ds in ALL_DATASETS:
    print(ds)
    # ds = 'm4_hourly'
    dataset = get_dataset(dataset_name=ds, regenerate=False)

    train = list(dataset.train)
    train_list = [pd.Series(ts['target'],
                            index=pd.date_range(start=ts['start'],
                                                freq=ts['start'].freq,
                                                periods=len(ts['target'])))
                  for ts in train]

    for i, series in enumerate(train_list):
        ds_size[f'{ds}_{i}.csv'] = len(series)

ds_size_df = pd.Series(ds_size).reset_index()
ds_size_df.columns = ['file', 'size']

pdiff_by_method = {}
for method in METHODS:
    pdiff = 100 * ((results[f'{method}+FTN(Alpha)'] - results[method]) / results[method])

    pdiff_by_method[method] = pdiff

pdiff_df = pd.concat(pdiff_by_method, axis=1).reset_index()
pdiff_df = pdiff_df.rename(columns={'index': 'file'})

df = pdiff_df.merge(ds_size_df, on='file')
df = df.drop('file', axis=1).melt('size')
df['logvalue'] = LogTransformation.transform(df['value'])

plot_sp = \
    ggplot(data=df,
           mapping=aes(x='size', y='logvalue')) + \
    geom_point(fill=MAIN_COLOR,
               color=MAIN_COLOR2,
               size=2) + \
    theme_classic(base_family='Palatino',
                  base_size=12) + \
    theme(plot_margin=.15,
          axis_text=element_text(size=14),
          axis_text_x=element_text(angle=0),
          strip_text=element_text(size=13),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_hline(yintercept=0,
               linetype='dashed',
               color='red',
               size=1) + \
    facet_wrap('~variable', ncol=1, scales='fixed') + \
    labs(y='Log % Diff', x='Data size')

plot_sp.save('results/plots/plot_lc_analysis.pdf', height=7, width=12)
