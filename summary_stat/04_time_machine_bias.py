from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus
from scipy.stats import ttest_ind


def apply_ttest(group):
    group1 = group[group['no_rel'] == 0]['abret_abs']
    group2 = group[group['no_rel'] == 1]['abret_abs']

    # Only perform t-test if both groups have data
    if not group1.empty and not group2.empty:
        t_stat, p_val = ttest_ind(group1, group2)
        return pd.Series({'t_stat': t_stat, 'p_val': p_val})
    else:
        return pd.Series({'t_stat': None, 'p_val': None})


def plot_abret_per_news_per_mkt_cap(df):
    # Create a 5x2 subplot structure
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    fig.tight_layout(pad=5.0)

    for i, mcap_d_val in enumerate(range(1, 11)):
        # Row and column indices for the subplot
        row = i // 2
        col = i % 2

        # Filter data and generate pivot table
        d = df.loc[df['mcap_d'] == mcap_d_val].groupby(['evttime', 'no_rel'])['abret_abs'].mean().reset_index()
        c = df.loc[(df['mcap_d'] == mcap_d_val)].groupby(['no_rel'])['abret_abs'].count()
        pivot_data = d.pivot(columns='no_rel', index='evttime', values='abret_abs')

        # Plotting on the appropriate subplot
        pivot_data.plot(ax=axes[row, col], title=f'mcap_d = {mcap_d_val}, 1/0 = {np.round(c[1]/c[0],3)}')


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/ss/04_time_machine_bias/eventff{window}/'
    os.makedirs(save_dir,exist_ok=True)

    per =data.load_some_relevance_icf(reload=False)

    ev = data.load_e_ff_long(window=window,reload=False).rename(columns={'evtdate':'adate'})

    ev=ev.sort_values(['permno','evttime']).reset_index(drop=True).dropna()

    ev['abret'] = PandasPlus.winzorize_series(ev['abret'],1)
    # ev['ret'] = winzorize_series(ev['ret'],1)

    ev['cabret']=ev.groupby('uid')['abret'].transform('cumsum')

    ev=ev.dropna()
    # l=ev.groupby(['permno','fdate'])['cabret'].last().sort_index()
    # s=ev_short.groupby(['permno','fdate'])['car'].last().sort_index()
    # l-s


    df = ev.merge(per)
    df['items_name'] = df['items'].map(Constant.ITEMS)
    df = df.dropna(subset='items_name')

    # first we find which events are pos and which are neg by looking at mean abret
    df = df.sort_values(['uid','evttime']).reset_index()
    temp = df.loc[(df['evttime']<=1) & (df['evttime']>=0),:].groupby('uid')['abret'].sum().reset_index().rename(columns={'abret':'sign'})
    temp['sign'] = (temp['sign']>0)*1
    df = df.merge(temp)

    df['abret_abs'] = df['abret'].abs()
    # df['cabret_abs'] = df['cabret'].abs()
    # df['abret_abs'] = df['abret']*((1-df['sign'])*-1)
    df['cabret_abs'] = df['cabret']*((1-df['sign'])*-1)

    df['hours'] = df['atime'].apply(lambda x: int(str(x)[:2]))


    d=df.groupby(['evttime','no_rel'])['abret_abs'].mean().reset_index().pivot(columns='no_rel',index='evttime',values='abret_abs')
    d.plot()
    plt.show()

    plot_abret_per_news_per_mkt_cap(df)
    plt.show()

    year = 2008
    plot_abret_per_news_per_mkt_cap(df.loc[df['year']==year])
    plt.show()


    plot_abret_per_news_per_mkt_cap(df.loc[(df['hours']<=15) & (df['hours']>=9)])
    plt.title(f'Year {year}')
    plt.show()


    plot_abret_per_news_per_mkt_cap(df.loc[(df['hours']>16) | (df['hours']<8)])
    plt.title(f'Year {year}')
    plt.show()



    list_of_items = ['cat','items','items_name','year','no_rel']
    temp = df.loc[df['evttime']>=0,:].groupby(list_of_items+['uid'])['abret_abs'].sum().reset_index()
    mean = temp.groupby(list_of_items)['abret_abs'].mean().reset_index()
    count = temp.groupby(list_of_items)['abret_abs'].count().reset_index()

    min_val = 100
    m = mean[count['abret_abs']>=min_val].reset_index()
    count = count[count['abret_abs']>=min_val].reset_index()

    count_for_attila = count.groupby(['items','no_rel'])['abret_abs'].sum()
    count_for_attila=count_for_attila.reset_index().pivot(columns='no_rel',index='items',values='abret_abs').fillna(f'<{min_val}')

    for cat in range(1, 10):
        print(cat)
        save_me = False
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # 2 subplots side by side
        for idx, no_news in enumerate([1, 0]):
            temp = m.loc[m['no_rel'] == no_news, :].pivot(columns='items', values='abret_abs', index='year')
            temp = temp.dropna(axis=1, how='any')
            temp = temp[[x for x in temp.columns if np.floor(x) == cat]]
            temp.columns = [str(x) + ' : ' + str(count_for_attila.loc[x] / 1e3) for x in temp.columns]

            if temp.shape[1]>0:
                save_me = True
                temp.plot(ax=axes[idx])  # Assign the corresponding subplot axis
                if no_news == 1:
                    axes[idx].set_title(f'No News On The Same Day')  # Optional title for each subplot
                    # plt.xlabel(f'Initial Reaction Positive')  # Optional title for each subplot
                else:
                    axes[idx].set_title(f'Some News On The Same Day')  # Optional title for each subplot
        # Add a subtitle annotation
        fig.text(0.5, 0.96, f'{Constant.SECTIONS[cat]}', ha="center", fontsize=12)  # Adjust the y value as needed
        plt.savefig(save_dir+f'{cat}.png')
        plt.tight_layout()
        plt.show()


    temp_dict = {}
    for idx, no_news in enumerate([1, 0]):
        temp = df.loc[df['no_rel'] == no_news, :].groupby(['evttime', 'items'])['cabret_abs'].mean().reset_index().pivot(columns='items', index='evttime', values='cabret_abs')
        count = df.loc[df['no_rel'] == no_news, :].groupby(['evttime', 'items'])['cabret_abs'].count().reset_index().pivot(columns='items', index='evttime', values='cabret_abs')
        col_to_keep = count.mean() > 1000
        temp_dict[idx] = temp[col_to_keep.index[col_to_keep.values]]

    for cat in range(1, 10):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # 2 subplots side by side
        for idx, no_news in enumerate([1, 0]):
            temp = temp_dict[idx].copy()


            temp = temp.dropna(axis=1, how='any')
            temp = temp[[x for x in temp.columns if np.floor(x) == cat]]
            # temp.columns = [str(x) + ' : ' + str(count_for_attila.loc[x] / 1e3) for x in temp.columns]

            if temp.shape[1]>0:
                save_me = True
                temp.plot(ax=axes[idx])  # Assign the corresponding subplot axis
                if no_news == 1:
                    axes[idx].set_title(f'No News On The Same Day')  # Optional title for each subplot
                    # plt.xlabel(f'Initial Reaction Positive')  # Optional title for each subplot
                else:
                    axes[idx].set_title(f'Some News On The Same Day')  # Optional title for each subplot
        # Add a subtitle annotation
        fig.text(0.5, 0.96, f'{Constant.SECTIONS[cat]}', ha="center", fontsize=12)  # Adjust the y value as needed
        plt.savefig(save_dir+f'{cat}.png')
        plt.tight_layout()
        plt.show()