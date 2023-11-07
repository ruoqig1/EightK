import gc

import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from itertools import chain
from utils_local.plot import plot_ev, plot_ev_no_conf
from matplotlib import pyplot as plt
from didipack import PlotPlus

def check_time_effect():
    ev = pd.read_pickle(data.p_dir + 'abn_ev_monly.p')
    ev['ym'] = PandasPlus.get_ym(ev['date'])
    ts  ='ym'
    temp = ev.loc[ev['evttime'].between(-1,1),:].groupby(['date','permno'])['abs_abret'].mean().reset_index()
    temp['year'] = temp['date'].dt.year
    temp['ym'] = PandasPlus.get_ym(temp['date'])
    high = temp.groupby(ts)['abs_abret'].mean()
    temp = ev.loc[~ev['evttime'].between(-1,1),:].groupby(['date','permno'])['abs_abret'].mean().reset_index()
    temp['year'] = temp['date'].dt.year
    temp['ym'] = PandasPlus.get_ym(temp['date'])
    low = temp.groupby(ts)['abs_abret'].mean()


    m = ((high-low)/low).reset_index()
    # m = ((high/low)).reset_index()
    temp = df.groupby(ts)['news'].mean()
    m['coverage'] = temp.values
    m=m.set_index(ts)
    m.index = pd.to_datetime(m.index,format='%Y%m')
    PlotPlus.plot_dual_axis(m,'abs_abret','coverage')
    plt.show()

def do_one_mona_lisa(df,to_label,label_begining ='', color='Blues'):
    cmap = plt.get_cmap(color)

    list_id = np.linspace(0.25, 1, df.shape[1])
    for i, c in enumerate(df.columns):
        if c in to_label:
            plt.plot(df.index, df[c], color=cmap(list_id[i]), label=f'{label_begining}{c}')
        else:
            plt.plot(df.index, df[c], color=cmap(list_id[i]))


if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    use_constant_in_abret = False
    save_dir = Constant.EMB_PAPER+'ss/'
    os.makedirs(save_dir, exist_ok=True)


    df = data.load_some_relevance_icf()
    df['news'] = df['no_rel']==0
    df=df.rename(columns={'adate':'date'})

    df['year'] = df['date'].dt.year
    df['ym'] = PandasPlus.get_ym(df['date'])
    df.groupby(['year','permno'])['form_id'].nunique().reset_index().groupby('year')['form_id'].mean().plot()
    plt.show()

    temp = df.groupby(['year','permno','mcap_d'])['form_id'].nunique().reset_index().groupby(['year','mcap_d'])['form_id'].mean().reset_index().pivot(columns='mcap_d',index='year',values='form_id')
    do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    plt.show()

    temp = df.groupby(['year','permno','mcap_d'])['news'].mean().reset_index().groupby(['year','mcap_d'])['news'].mean().reset_index().pivot(columns='mcap_d',index='year',values='news')
    do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    plt.show()

    rav = data.load_ravenpack_all()
    rav['year'] = rav['rdate'].dt.year
    rav['ym'] = PandasPlus.get_ym(rav['rdate'])
    temp = rav.groupby('ym')['permno'].count()
    temp.plot()
    plt.show()



    temp=pd.DataFrame(temp)
    temp['coverage'] = df.groupby('ym')['news'].mean()
    temp = temp.dropna()
    temp.index = pd.to_datetime(temp.index,format='%Y%m')
    temp = temp.rolling(12).mean()
    PlotPlus.plot_dual_axis(temp,'permno','coverage')
    plt.show()
    temp.corr()

    # do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    # plt.show()