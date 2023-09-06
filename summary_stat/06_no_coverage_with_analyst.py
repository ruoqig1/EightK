from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus
from scipy.stats import ttest_ind
from utils_local.plot import *
import statsmodels.api as sm
from didipack import OlsPLus



if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    reg = OlsPLus()

    save_dir = Constant.DROP_RES_DIR + f'/ss/05_no_coverage/'
    os.makedirs(save_dir,exist_ok=True)

    df = data.load_main_no_rel_data()

    ibes = data.load_analyst().rename(columns={'rdate':'date'})

    df = df.merge(ibes,how='left')
    df['some_analyst'] = (~pd.isna(df['rec_value']))*1
    df['some_news'] = (df['relevance']>80)*1
    df['some_analyst*no_news'] = (1-df['some_news'])*df['some_analyst']


    d=df.groupby('year')['some_analyst'].mean()
    d.plot()
    plt.ylabel('8k With some Analyst Rec')
    plot_with_every_ticks_tight_and_grid(d.index)
    plt.show()


    d=df.groupby('items')['some_analyst*some_news'].mean()
    plt.ylabel('8k With some Analyst Rec and No News')
    d.plot()
    plot_with_every_ticks_tight_and_grid(d.index)
    plt.show()

    for c in ['some_analyst', 'some_analyst*some_news']:
        d=df.groupby('items')[c].mean()
        d = pd.DataFrame(d)
        big_items_by_items_plots(d)
        plt.ylabel(c)
        plt.show()

    plot_abret_per_news_per_mkt_cap(df.loc[df['no_rel']==1,:],group_col_name='some_analyst')
    plt.show()


    # fe = ['mcap_d']
    # x = ['some_news','some_analyst']
    # x = ['some_analyst','some_news','some_analyst*no_news']
    # y = ['abret_abs']
    # cluster_col =['permno']
    # temp=df.loc[df['evttime']==0,y+x+fe+cluster_col].dropna()
    #
    #
    # m=reg.sm_with_fixed_effect(temp,y=y[0],x=x,fix_effect_cols=fe, std_error_cluster=cluster_col)
    # m.summary()



    fe = ['mcap_d']
    x = ['some_news','some_analyst']
    x = ['some_analyst','some_news','some_analyst*no_news']
    y = ['abret_abs']
    cluster_col =['permno']
    temp=df.loc[df['evttime']==0,y+x+fe+cluster_col].dropna()
    # temp=temp.loc[temp['mcap_d']<=8,:]
    temp=temp.loc[temp['mcap_d']==10,:]
    temp['one']=1

    m = sm.OLS(temp[y], temp[x+['one']]).fit(cov_type='cluster', cov_kwds={'groups': temp[cluster_col]})
    m.summary()
