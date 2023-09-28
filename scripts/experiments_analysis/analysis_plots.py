import re

from plotnine import *
import numpy as np
import pandas as pd

from src.results.reading_files import ReadFiles
from src.preprocessing.log import LogTransformation
from config import MAIN_COLOR, MAIN_COLOR2

METHODS = ['ARIMA', 'Naive', 'Theta',
           'Ensemble', 'FTNE',
           'DirRec', 'Recursive', 'Direct', 'KNN']

METHODS_FTN = [f'{x}+FTN' for x in METHODS]
METHODS_SMOOTH = [f'{x}+FTN(Smooth)' for x in METHODS]
METHODS_ALPHA = [f'{x}+FTN(Alpha)' for x in METHODS]

results = ReadFiles.read_all_files(horizon=None)

# AVERAGE RANK
avg_rank = results.loc[:, METHODS + METHODS_ALPHA].rank(axis=1).mean().sort_values()

avg_rank_df = avg_rank.reset_index()
avg_rank_df.columns = ['Method', 'Avg. Rank']
avg_rank_df['Method'] = [re.sub('\(Smooth\)', '',x) for x in avg_rank_df['Method']]
avg_rank_df['Method'] = [re.sub('\(Alpha\)', '',x) for x in avg_rank_df['Method']]
avg_rank_df['Method'] = pd.Categorical(avg_rank_df['Method'], categories=avg_rank_df['Method'].values[::-1])

plot_avgrank = \
    ggplot(data=avg_rank_df,
           mapping=aes(x='Method', y='Avg. Rank')) + \
    geom_bar(position='dodge',
             stat='identity',
             width=0.8,
             fill=MAIN_COLOR) + \
    theme_classic(base_family='Palatino',
                  base_size=12) + \
    theme(plot_margin=.2,
          axis_text_x=element_text(size=12, angle=0),
          axis_text_y=element_text(size=14),
          legend_title=element_blank()) + \
    labs(x='') + coord_flip()

# AVERAGE RANK BY FTN

avg_rank_df['Type'] = [bool(re.search('FTN$', x)) for x in avg_rank_df['Method']]
avg_rank_df['Type'] = avg_rank_df['Type'].map({True: 'With FTN', False: 'Without FTN'})
avg_rank_df['BaseMethod'] = [re.sub('\+FTN', '', x) for x in avg_rank_df['Method']]

plot_avgrankftn = \
    ggplot(data=avg_rank_df,
           mapping=aes(x='BaseMethod',
                       y='Avg. Rank',
                       group='Type',
                       fill='Type')) + \
    theme_classic(base_family='Palatino',
                  base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          strip_text=element_text(size=13),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_bar(position='dodge', stat='identity') + \
    labs(x='') + \
    scale_fill_manual(values=['#3e849e', '#a42838'])

# PERCENTAGE DIFFERENCE BY FTN

pairwise_err_df = {}
for method in METHODS:
    method_ftn = f'{method}+FTN(Alpha)'
    pd_perf = 100 * ((results[method_ftn] - results[method]) / results[method])

    pairwise_err_df[method] = pd_perf

pairwise_err_df = pd.DataFrame(pairwise_err_df)

pairwise_err_df_m = pairwise_err_df.melt()
pairwise_err_df_m['logvalue'] = LogTransformation.transform(pairwise_err_df_m['value'])

plot_pdftn = \
    ggplot(data=pairwise_err_df_m,
           mapping=aes(x='variable', y='logvalue')) + \
    geom_boxplot(fill=MAIN_COLOR2) + \
    theme_classic(base_family='Palatino',
                  base_size=12) + \
    theme(plot_margin=.125,
          axis_text=element_text(size=14),
          strip_text=element_text(size=13),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_hline(yintercept=0,
               linetype='dashed',
               color='red',
               size=1) + \
    labs(y='Log % Diff', x='')

# PERCENTAGE DIFFERENCE FROM RECURSIVE
results_subset = results.loc[:, METHODS + METHODS_ALPHA]
pd_to_rec_df, ref = {}, 'Recursive'
for c in results_subset:
    if c == 'Recursive':
        continue

    pd_to_rec_df[c] = 100 * ((results_subset[c] - results_subset[ref]) / results_subset[ref])

pd_to_rec_df = pd.DataFrame(pd_to_rec_df)
pd_to_rec_df.columns = [re.sub('\(Alpha\)', '',x) for x in pd_to_rec_df.columns]
ord = pd_to_rec_df.median().sort_values().index.values[::-1]

pd_to_rec_df_m = pd_to_rec_df.melt()
pd_to_rec_df_m['logvalue'] = LogTransformation.transform(pd_to_rec_df_m['value'])
pd_to_rec_df_m['variable'] = pd.Categorical(pd_to_rec_df_m['variable'], categories=ord)


plot_pdrec = \
    ggplot(data=pd_to_rec_df_m,
           mapping=aes(x='variable', y='logvalue')) + \
    geom_boxplot(fill=MAIN_COLOR2) + \
    theme_classic(base_family='Palatino',
                  base_size=12) + \
    theme(plot_margin=.2,
          axis_text=element_text(size=14),
          axis_text_x=element_text(angle=0),
          strip_text=element_text(size=13),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_hline(yintercept=0,
               linetype='dashed',
               color='red',
               size=1) + \
    labs(y='Log % Diff', x='') + coord_flip()

# EVENT PROBABILITIES
ROPE = 1


def comp_probs(x):
    left = (x < -ROPE).mean()
    right = (x > ROPE).mean()
    mid = np.mean([-ROPE < x_ < ROPE for x_ in x])

    return left, mid, right


df = pairwise_err_df.apply(comp_probs, axis=0).T.reset_index()
df.columns = ['Method', 'FTN wins', 'draw', 'FTN loses']
df_melted = df.melt('Method')
df_melted['variable'] = pd.Categorical(df_melted['variable'],
                                       categories=['FTN loses', 'draw', 'FTN wins'])

bayes_plot = \
    ggplot(df_melted, aes(fill='variable', y='value', x='Method')) + \
    geom_bar(position='stack', stat='identity') + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.175,
          axis_text=element_text(size=12),
          strip_text=element_text(size=14),
          axis_text_x=element_text(size=14, angle=0),
          legend_title=element_blank(),
          legend_position='top') + \
    labs(x='', y='Proportion of probability') + \
    scale_fill_hue()

# AVG RANK DIFF BY HORIZON


MAX_FH = 18
pd_over_fh = []
for fh_ in range(MAX_FH):
    print(fh_)
    # res_fh = ReadFiles.read_all_files(horizon=[x for x in range(fh_)])
    res_fh = ReadFiles.read_all_files(horizon=[fh_])

    pairwise_err_df = {}
    for method in METHODS:
        # method='DirRec'
        print(method)
        method_ftn = f'{method}+FTN(Smooth)'
        ar = res_fh[[method, method_ftn]].rank(axis=1).mean()

        ar_dt = ar[method_ftn] - ar[method]

        pairwise_err_df[method] = ar_dt

    pairwise_err_df = pd.Series(pairwise_err_df)

    pd_over_fh.append(pairwise_err_df)

pd_over_fh_df = pd.DataFrame(pd_over_fh)
pd_over_fh_df_ind = pd_over_fh_df.reset_index()
pd_over_fh_df_ind['index'] += 1
# pd_over_fh_df_ind = pd_over_fh_df_ind[1:]

plot_fh = \
    ggplot(data=pd_over_fh_df_ind.melt('index'),
           mapping=aes(x='index',
                       y='value',
                       color='variable',
                       group='variable')) + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.175,
          axis_text=element_text(size=12),
          strip_text=element_text(size=14),
          axis_text_x=element_text(size=12, angle=0),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    geom_point(group='variable') + \
    geom_hline(yintercept=0,
               linetype='dashed',
               color='black',
               size=1.5) + \
    labs(x='Forecasting Horizon',
         y='Diff. in Avg Rank')

print(plot_fh)

# plot_avgrank.save('plot_avgrank.pdf', height=5, width=12)
# plot_avgrankftn.save('plot_avgrankftn.pdf', height=5, width=12)
# plot_pdftn.save('plot_pdftn.pdf', height=5, width=12)
# plot_pdrec.save('plot_pdrec.pdf', height=7, width=12)
# bayes_plot.save('plot_bayes.pdf', height=5, width=12)
# plot_fh.save('plot_fh.pdf', height=5, width=12)
