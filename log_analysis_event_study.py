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

    temp = df.loc[:,:].groupby(['dist','permno','news0','year'])['ip'].mean().reset_index()
    temp = temp.pivot(columns='news0',index=['dist','permno','year'],values='ip')
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
        temp.loc[temp['mcap_d'] == d, :].groupby('dist')['covered_r'].median().plot(ax=ax)
        ax.set_title(f'MCAP d = {d}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Median Ratio')
        ax.grid()
    plt.tight_layout()
    plt.savefig(save_dir+'logs_days_per_mcap_yearly_sytem.png')
    plt.show()


    temp.loc[:, :].groupby('dist')['covered_r'].median().plot()
    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Median Ratio')
    plt.tight_layout()
    plt.savefig(save_dir+'logs_day_yearly_system.png')
    plt.show()



    temp.loc[temp['dist']<41, :].groupby('dist')[[0.0,1.0]].mean().cumsum().plot()
    plt.title(f'MCAP all')
    plt.grid()
    plt.show()




    temp = df.loc[:,:].groupby(['dist','permno','news0'])['ip'].mean().reset_index()
    temp = temp.pivot(columns='news0',index=['dist','permno'],values='ip')
    temp['covered_r'] = temp[1.0]/temp[0.0]
    # temp['covered_r'] = temp[0.0]/temp[1.0]
    temp = temp.dropna()
    temp = temp.reset_index()

    temp.loc[:, :].groupby('dist')['covered_r'].median().plot()
    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Median Covered Ratio')
    plt.tight_layout()
    plt.savefig(save_dir+'log_days_median_ev.png')
    plt.show()






    df = data.load_logs_ev_study()

    ati = data.load_icf_ati_filter()
    ati = ati.rename(columns={'date':'form_date'})
    ati = ati.loc[ati['items'].isin(Constant.LIST_ITEMS_TO_USE),['form_date','form_id','news0','permno','items']].drop_duplicates()
    df = df.merge(ati)
    cov_per_item = df.groupby(['items','news0'])['ip'].count().reset_index().pivot(columns='news0',index='items',values='ip').fillna(0.0)
    # Your existing data processing code remains the same.

    # Define the subplot grid
    fig, axs = plt.subplots(5, 4, figsize=(20, 25))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten the axes array for easy indexing

    # Iterate over items and plot
    i = 0
    for _, items in enumerate(np.sort(np.unique(df['items']))):
        if cov_per_item.loc[items, :].min() > 2000:
            temp = df.loc[df['items'] == items, :].groupby(['dist', 'permno', 'news0'])['ip'].mean().reset_index()
            temp = temp.pivot(columns='news0', index=['dist', 'permno'], values='ip')
            temp['covered_r'] = temp[1.0] / temp[0.0]
            temp = temp.dropna()
            temp = temp.reset_index()

            # Plot on the respective subplot
            temp.groupby('dist')['covered_r'].median().plot(ax=axs[i])
            axs[i].set_title(f'Items {items}')
            axs[i].grid()
            axs[i].set_xlabel('Days')
            axs[i].set_ylabel('Median Covered Ratio')
            plt.tight_layout()
            i+=1

    # Show the entire grid of plots
    plt.tight_layout()

    plt.savefig(save_dir + 'log_days_per_items_ev.png')
    plt.show()







