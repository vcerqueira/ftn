from plotnine import *
import pandas as pd

from src.results.reading_files import ReadFiles

METHODS = ['DirRec', 'Recursive', 'Direct']

METHODS_FTN = [f'{x}+FTN' for x in METHODS]
METHODS_SMOOTH = [f'{x}+FTN(Smooth)' for x in METHODS]
METHODS_ALPHA = [f'{x}+FTN(Alpha)' for x in METHODS]

ALL_METHODS = METHODS + METHODS_ALPHA + METHODS_SMOOTH + METHODS_FTN

results = ReadFiles.read_all_files(horizon=None)

# AVERAGE RANK
avg_rank = results.loc[:, ALL_METHODS].rank(axis=1).mean().sort_values()

avg_rank_df = avg_rank.reset_index()
avg_rank_df.columns = ['Method', 'Avg. Rank']
avg_rank_df['Method'] = pd.Categorical(avg_rank_df['Method'], categories=avg_rank_df['Method'].values[::-1])


avg_rank_df['Type'] = [
    'FTN Alpha',
    'FTN Alpha',
    'FTN Smooth',
    'No FTN',
    'FTN Alpha',
    'FTN Smooth',
    'FTN',
    'FTN Smooth',
    'FTN',
    'FTN',
    'No FTN',
    'No FTN',
]

avg_rank_df['BaseMethod'] = [
    'Direct',
    'DirRec',
    'Direct',
    'Direct',
    'Recursive',
    'DirRec',
    'Direct',
    'Recursive',
    'DirRec',
    'Recursive',
    'DirRec',
    'Recursive',
]

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
    labs(x='')

plot_avgrankftn.save('results/plots/ftnvariants.pdf', height=5, width=12)
