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
    window = 20

    save_dir = Constant.DROP_RES_DIR + f'/ss/01_event_study_aggregated/eventff{window}/'
    os.makedirs(save_dir,exist_ok=True)

    per =data.load_list_by_date_time_permno_type()

    # create a few variation of abnormal returns
    ev = data.load_e_ffshort(window=window)
    ev['cret_abs'] = ev['cret'].abs()
    ev['car_abs'] = ev['car'].abs()
    ev['car_pos'] = ev['car']
    ev.loc[ev['car']<0,'car_pos'] = np.nan
    ev['car_neg'] = ev['car']
    ev.loc[ev['car']>0,'car_neg'] = np.nan
    col_of_int = ['car_abs','car_pos','car_neg']
    col_of_int = ['car_abs']

    # winzorise our column of interest
    ev[col_of_int[0]] = winzorize_series(ev[col_of_int[0]],1)

    # merge with all dates
    df = per[['permno','fdate']].drop_duplicates().merge(ev)

    df.mean()
    df.groupby(df['fdate'].dt.year)[col_of_int].mean().plot()
    plt.ylabel('abs(car)')
    plt.title('Average abs across all items')
    plt.tight_layout()
    plt.savefig(save_dir+'avg_across_all_items')
    plt.show()


    per['items_name']=per['items'].map(Constant.ITEMS)
    ind=per.groupby(['permno','fdate'])['items'].transform('nunique')

    t=per.loc[:,:].dropna(subset=['items_name']).merge(ev).groupby(['items','items_name'])[col_of_int].aggregate(['mean','count']).sort_values(('car_abs','mean'))
    t.to_csv(save_dir+'category_carr.csv')
    for i in range(1,3):
        t=per.loc[ind<=i,:].dropna(subset=['items_name']).merge(ev).groupby(['items','items_name'])[col_of_int].aggregate(['mean','count']).sort_values(('car_abs','mean'))
        t.to_csv(save_dir+f'category_carr_n{i}.csv')

    t = per.dropna(subset=['items_name']).merge(ev).groupby(['items', 'items_name','year'])[col_of_int].aggregate(['mean', 'count']).sort_values(('car_abs', 'mean'))
    t=t.reset_index()
    t.columns = [(col[0] if col[1] == '' else col[1]) for col in t.columns]
    t['cat'] = np.floor(t['items'])
    min_count = 50
    min_count_total = 50
    for c in t['cat'].unique():
        df = t.loc[t['cat']==c,:]
        df = df.loc[df['count']>=min_count,:]
        ind = df.groupby('items')['count'].transform('mean')>=min_count_total
        df = df.loc[ind,:]
        df=df.pivot(index='year',columns='items',values='mean')
        if df.shape[0]>0:
            df.plot()
            plt.title(Constant.SECTIONS[c])
            plt.tight_layout()
            plt.savefig(save_dir+f'section_{int(c)}.png')
            plt.show()

    df = per.dropna(subset=['items_name'])

    # Pivot DataFrame
    pivot_df = df.groupby(['permno', 'fdate', 'items']).size().unstack(fill_value=0)
    pivot_df[pivot_df > 1] = 1  # If there are multiple occurrences of the same item, set them to 1


    # Compute probabilities
    P_B = pivot_df.mean()
    P_A_and_B = pivot_df.T.dot(pivot_df) / len(pivot_df)

    # Use broadcasting for division and prevent division by zero using `.replace`
    P_A_given_B = (P_A_and_B.div(P_B, axis='columns')).replace([np.inf, -np.inf], np.nan)

    # Diagonal values will not make sense as they'll represent P(itemX | itemX), so set them to NaN
    for item in P_A_given_B.columns:
        P_A_given_B.at[item, item] = np.nan

    print(P_A_given_B)

    # Plot heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(P_A_given_B, annot=True, cmap="YlGnBu", cbar_kws={'label': 'P(A|B)'})
    plt.title("Conditional Probabilities Heatmap")
    plt.ylabel('Item A')
    plt.xlabel('Given Item B')
    plt.tight_layout()
    plt.savefig(save_dir+'prob_of_items_given_items.png')
    plt.show()
