from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from parameters import *


def apply_ttest(group,group_col ='no_rel',y_col ='abret_abs'):
    group1 = group[group[group_col] == 0][y_col]
    group2 = group[group[group_col] == 1][y_col]

    # Only perform t-test if both groups have data
    if not group1.empty and not group2.empty:
        t_stat, p_val = ttest_ind(group1, group2)
        return pd.Series({'t_stat': t_stat, 'p_val': p_val})
    else:
        return pd.Series({'t_stat': None, 'p_val': None})

def big_items_by_items_plots(d):
    # Plotting
    plt.figure(figsize=(15, 8))
    for column in d.columns:
        plt.plot([str(x) for x in d.index], d[column], marker='o', label=column)

    plt.xticks(ticks=[str(x) for x in d.index], labels=[str(x) for x in d.index], rotation=45)
    plt.xlabel('Items')
    plt.ylabel('Value')
    plt.title('Plot of dataframe')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()




def plot_abret_per_news_per_mkt_cap(df,group_col_name='no_rel'):
    # Create a 5x2 subplot structure
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    fig.tight_layout(pad=5.0)

    for i, mcap_d_val in enumerate(range(1, 11)):
        # Row and column indices for the subplot
        row = i // 2
        col = i % 2

        # Filter data and generate pivot table
        d = df.loc[df['mcap_d'] == mcap_d_val].groupby(['evttime', group_col_name])['abret_abs'].mean().reset_index()
        c = df.loc[(df['mcap_d'] == mcap_d_val)].groupby([group_col_name])['abret_abs'].count()
        pivot_data = d.pivot(columns=group_col_name, index='evttime', values='abret_abs')

        # Plotting on the appropriate subplot
        pivot_data.plot(ax=axes[row, col], title=f'mcap_d = {mcap_d_val}, 1/0 = {np.round(c[1]/c[0],3)}')

def plot_with_every_ticks_tight_and_grid(index):
    plt.xticks(index,index,rotation=90)
    plt.grid()
    plt.tight_layout()