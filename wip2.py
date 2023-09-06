from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, PlotPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    df = data.load_main_no_rel_data(False)
    df = df.loc[df['items'].isin(Constant.LIST_ITEMS_TO_USE),:]
    df.columns
    press=data.get_press_release_bool_per_event()
    press.dtypes
    df=df.merge(press)

    ind = df['date'].dt.year>2004
    df=df.loc[ind,:]


    ind = (df['evttime'].abs()<=10) &(df['mcap_d']>=3)
    df = df.loc[ind,:]

    df.dtypes
    df['news'] = df['no_rel']==0


    df['release']*=1
    df['large_firm'] = (df['mcap_d']==10)*1
    df.head()
    #

    for large_firm in [1, 0]:
        for press in [1, 0]:
            for news in [1,0]:
                ind = (df['news']==news) & (df['large_firm']==large_firm) & (df['release']==press)
                # temp=df.groupby(['evttime','news','large_firm','release'])['abs_abret'].mean().reset_index()
                temp=df.loc[ind,:].groupby(['evttime'])['abret_abs'].mean()
                plt.plot(temp.index,temp.values,label=f'News {news}, Press Release {press}')
        plt.legend()
        plt.grid()
        plt.title(f'Large Firm {large_firm}')
        plt.tight_layout()
        plt.show()

    for press in [1, 0]:
        for news in [1,0]:
            ind = (df['news']==news) & (df['release']==press)
                # temp=df.groupby(['evttime','news','large_firm','release'])['abs_abret'].mean().reset_index()
            temp=df.loc[ind,:].groupby(['evttime'])['abret_abs'].mean()
            plt.plot(temp.index,temp.values,label=f'News {news}, Press Release {press}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



    df.groupby('release')['news'].mean()