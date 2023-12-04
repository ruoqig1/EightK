import didipack as didi
from matplotlib import pyplot as plt
from data import Data
from parameters import *

if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    save_dir = Constant.EMB_PAPER
    data = Data(par)
    df = data.load_logs_ev_study()

    ati = data.load_icf_ati_filter()
    ati = ati.rename(columns={'date':'form_date'})
    ati = ati.loc[ati['items'].isin(Constant.LIST_ITEMS_TO_USE),['form_date','form_id','news0','permno']].drop_duplicates()
    df = df.merge(ati)
    df['year'] = df['form_date'].dt.year

    df['c_ip'] = df.groupby(['permno','form_date'])['ip'].cumsum()

    temp = df.loc[:,:].groupby(['dist','permno','news0','year'])['c_ip'].mean().reset_index()
    temp = temp.pivot(columns='news0',index=['dist','permno','year'],values='c_ip')
    temp['covered_r'] = temp[1.0]/temp[0.0]
    temp = temp.dropna()
    temp = temp.reset_index()
    temp = temp.merge(data.load_mkt_cap_yearly())


    n_rows = 5
    n_cols = 2
    # Create the subplot structure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 20))  # Adjust figsize as needed
    axes = axes.flatten()  #

    mcap_d_values = np.sort(temp['mcap_d'].unique())


    for i, d in enumerate(mcap_d_values):
        ax = axes[i]
        temp.loc[(temp['mcap_d'] == d) & (temp['dist']<41), :].groupby('dist')[[0.0,1.0]].median().plot(ax=ax)
        ax.set_title(f'MCAP d = {d}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Median Ratio')
        ax.grid()
    plt.tight_layout()
    plt.savefig(save_dir+'logs_days_per_mcap_yearly_sytem.png')
    plt.show()


    temp.loc[temp['dist']<41, :].groupby('dist')[[0.0,1.0]].median().plot()
    plt.title(f'MCAP all')
    plt.grid()
    plt.show()








