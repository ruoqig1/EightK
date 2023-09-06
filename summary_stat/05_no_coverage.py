from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus
from scipy.stats import ttest_ind
from utils_local.plot import *





if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/ss/05_no_coverage/'
    os.makedirs(save_dir,exist_ok=True)

    df = data.load_main_no_rel_data()


    # Check number of news in our sample
    rav = data.load_ravenpack_all()
    rav=rav.groupby(rav['rdate'].dt.year)['permno'].count()
    rav.plot()
    plt.ylabel('Nb News In ravenpack Per year')
    plot_with_every_ticks_tight_and_grid(rav.index)
    plt.savefig(save_dir+'nb_news_ravenpack')
    plt.show()



    d=df.groupby(['year','mcap_d'])['abret'].count().reset_index().pivot(columns='mcap_d',index='year',values='abret')
    d.plot()
    plt.title('Number of 8k per Year')
    plot_with_every_ticks_tight_and_grid(d.index)
    plt.savefig(save_dir+'nb_eightk_per_year')
    plt.show()

    d=df.groupby(['year','mcap_d'])['no_rel'].mean().reset_index().pivot(columns='mcap_d',index='year',values='no_rel')
    d.plot()
    plt.ylabel('Percentage of 8k NOT in the news')
    plot_with_every_ticks_tight_and_grid(d.index)
    plt.savefig(save_dir+'perc_not_cover_per_mkt_cap_and_year')
    plt.show()

    df.groupby(['year','mcap_d'])['permno'].nunique().reset_index().pivot(columns='mcap_d',index='year',values='permno').plot()
    plt.ylabel('Nb Firm')
    plot_with_every_ticks_tight_and_grid(d.index)
    plt.savefig(save_dir+'nb_firm')
    plt.show()


    plot_abret_per_news_per_mkt_cap(df)
    plt.tight_layout()
    plt.savefig(save_dir+'grid_main')
    plt.show()

    # check that it's robust outside of market hours
    plot_abret_per_news_per_mkt_cap(df.loc[(df['hours']<=16)])
    plt.savefig(save_dir+'grid_in_market_hours')
    plt.show()

    # and in market hours
    plot_abret_per_news_per_mkt_cap(df.loc[(df['hours']>17)])
    plt.savefig(save_dir+'grid_after_market_hours')
    plt.show()



    d0 = df.loc[(df['evttime'] == 0) & (df['no_rel']==0)].groupby(['large_cap','year'])['abret_abs'].mean().reset_index().pivot(columns='large_cap',index='year',values='abret_abs')
    d1 = df.loc[(df['evttime'] == 0) & (df['no_rel']==1)].groupby(['large_cap','year'])['abret_abs'].mean().reset_index().pivot(columns='large_cap',index='year',values='abret_abs')
    (d0-d1).plot()
    plt.title('Mean diff between mean abs abret on day 0')
    plot_with_every_ticks_tight_and_grid(d0.index)
    plt.savefig(save_dir+'ts_mean_diff_mabsabret_day_zero')
    plt.show()

    d0 = df.loc[(df['evttime'] == 0) & (df['no_rel']==0)].groupby(['items','large_cap'])['abret_abs'].mean().reset_index().pivot(columns='items',index='large_cap',values='abret_abs')
    d1 = df.loc[(df['evttime'] == 0) & (df['no_rel']==1)].groupby(['items','large_cap'])['abret_abs'].mean().reset_index().pivot(columns='items',index='large_cap',values='abret_abs')
    d = (d0-d1)

    dcount = df.loc[(df['evttime'] == 0) & (df['no_rel']==0)].groupby(['items','mcap_d'])['abret_abs'].count().reset_index().pivot(columns='items',index='mcap_d',values='abret_abs')
    dcount=dcount.loc[10,:]
    d=d[dcount[dcount>1000].index].T

    big_items_by_items_plots(d)
    plt.title('Mean diff between mean abs abret on day 0, across items')
    plt.savefig(save_dir+'cs_mean_diff_mabsabret_day_zero')
    plt.show()


    count = df.groupby('items')['abret'].count()
    count=count[count>1000*20].index

    d=df.groupby(['large_cap','items'])['no_rel'].mean().reset_index().pivot(columns='large_cap',index='items',values='no_rel')
    d=d.dropna().loc[count,:]
    big_items_by_items_plots(d)
    plt.ylabel('% Of Items Not Covered In the News')
    plt.savefig(save_dir+'cs_percentage_not_covered_in_the_news_per_item')
    plt.show()


    d=df.loc[df['no_rel']==0,:].groupby(['large_cap','items'])['abret_abs'].mean().reset_index().pivot(columns='large_cap',index='items',values='abret_abs')
    d=d.dropna().loc[count,:]
    big_items_by_items_plots(d)
    plt.ylabel('Abnormal Return Of 8k covered in the news')
    plt.savefig(save_dir+'cs_abret_on_day_zero_per_item')
    plt.show()