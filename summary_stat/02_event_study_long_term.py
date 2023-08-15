from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns


def winzorize_series(s, p=5):
    """
    Winzorizes a pandas Series by p percent on both sides of the distribution.

    Parameters:
    - s: The pandas Series to winzorize.
    - p: The percentage to winzorize on both sides. Defaults to 5%.

    Returns:
    - The winzorized Series.
    """
    lower = s.quantile(p / 100)
    upper = s.quantile(1 - p / 100)

    return s.clip(lower=lower, upper=upper)


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/ss/02_event_study_long_term/eventff{window}/'
    os.makedirs(save_dir,exist_ok=True)

    per =data.load_list_by_date_time_permno_type()
    per['items_name'] = per['items'].map(Constant.ITEMS)

    ev = data.load_e_ff_long(window=window,reload=False).rename(columns={'evtdate':'fdate'})

    ev=ev.sort_values(['permno','evttime']).reset_index(drop=True).dropna()

    ev['abret'] = winzorize_series(ev['abret'],1)
    # ev['ret'] = winzorize_series(ev['ret'],1)

    ev['cabret']=ev.groupby('uid')['abret'].transform('cumsum')

    ev=ev.dropna()
    # l=ev.groupby(['permno','fdate'])['cabret'].last().sort_index()
    # s=ev_short.groupby(['permno','fdate'])['car'].last().sort_index()
    # l-s


    df = ev.merge(per)

    df = df.dropna(subset='items_name')



    # first we find which events are pos and which are neg by looking at mean abret
    df = df.sort_values(['uid','evttime']).reset_index()
    temp = df.loc[(df['evttime']<=1) & (df['evttime']>=0),:].groupby('uid')['abret'].sum().reset_index().rename(columns={'abret':'sign'})
    temp['sign'] = (temp['sign']>0)*1
    df = df.merge(temp)

    mean = df.groupby(['items','evttime','sign'])['cabret'].mean()
    count = df.groupby(['items','evttime','sign'])['cabret'].count()
    m = mean[count>=1000].reset_index()


    count_for_attila = df.groupby(['items'])['cabret'].count()
    count_for_attila

    for cat in range(1, 10):
        print(cat)
        save_me = False
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # 2 subplots side by side
        for idx, pos_index in enumerate([1, 0]):
            temp = m.loc[m['sign'] == pos_index, :].pivot(columns='items', values='cabret', index='evttime')
            temp = temp.dropna(axis=1, how='any')
            temp = temp[[x for x in temp.columns if np.floor(x) == cat]]
            temp.columns = [str(x)+ ' : ' +str(count_for_attila.loc[x]/1e3) for x in temp.columns]



            if temp.shape[1]>0:
                save_me = True
                temp.plot(ax=axes[idx])  # Assign the corresponding subplot axis
                if pos_index == 1:
                    axes[idx].set_title(f'Initial Reaction Positive')  # Optional title for each subplot
                    # plt.xlabel(f'Initial Reaction Positive')  # Optional title for each subplot
                else:
                    axes[idx].set_title(f'Initial Reaction Negative')  # Optional title for each subplot
        # Add a subtitle annotation
        fig.text(0.5, 0.96, f'{Constant.SECTIONS[cat]}', ha="center", fontsize=12)  # Adjust the y value as needed
        plt.savefig(save_dir+f'{cat}.png')
        plt.tight_layout()
        plt.show()
