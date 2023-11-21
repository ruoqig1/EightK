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
import didipack as didi
from data import Data
from didipack import PandasPlus

def print_size(df,step):
    print(f'### Step {step}')
    print(df.groupby('alert')['body'].count()/1e6,flush=True)

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    nb_factors = 6
    if nb_factors == 6:
        save_dir = Constant.EMB_PAPER+'ss/'
    else:
        save_dir = Constant.EMB_PAPER+'temp/'
    os.makedirs(save_dir, exist_ok=True)



    df = data.load_some_relevance_icf()
    df['news'] = df['no_rel']==0
    df=df.rename(columns={'adate':'date'})
    df['big'] = df['mcap_d']==10
    df = df.loc[df['items'].isin(Constant.LIST_ITEMS_TO_USE),:]


    ev =data.load_abn_return(model=nb_factors)
    df = df.merge(ev)


    df = df[['date','permno','abs_abret','sigma_abs_ra','sigma_ra','news','big','mcap_d','evttime']].drop_duplicates()
    df['big'] = 1*(df['mcap_d']>=10) + 1*(df['mcap_d']>3)+ 1*(df['mcap_d']>6)
    df['big'] = 'micro'
    df.loc[df['mcap_d']<=10,'big'] = 'Mega'
    df.loc[df['mcap_d']<10,'big'] = 'Large'
    df.loc[df['mcap_d']<6,'big'] = 'Small'
    df.loc[df['mcap_d']<3,'big'] = 'Micro'
    sizes = ['Mega','Large','Small','Micro']


    sp = data.load_snp_const(False)
    sp['ym'] = PandasPlus.get_ym(sp['date'])
    sp = sp.drop(columns='date')
    df['ym'] = PandasPlus.get_ym(df['date'])
    df = df.merge(sp, how='left')

    df['in_snp'] = (((df['date'] <= df['ending']) & (df['date'] >= df['start'])) * 1).fillna(0.0)
    # df['big'] = df['in_snp'].replace({1:'snp',0:'non_snp'})
    # sizes =df['big'].unique()

    t = df.loc[df['evttime']==df['evttime'].min(),:].groupby(['date', 'permno'])['abs_abret'].mean().reset_index().rename(columns={'abs_abret':'norm'})
    df =df.merge(t)
    if nb_factors>0:
        df['abs_abret_norm'] = df['abs_abret'] -df['norm']
        df = df.loc[df['evttime'].between(-15, 15),:]
    else:
        df['abs_abret_norm'] = df['abs_abret']
    


    # df.groupby('big')['mcap_d'].unique()
    big = 'Mega'
    for big in sizes:
        if big =='Mega':
            plt.figure(figsize=[6.4*2,4.8])
            ind = (df['big'] == big) & (df['evttime'].between(-20, 20))  # & (df['mcap_d']>2)
        else:
            ind = (df['big'] == big) & (df['evttime'].between(-10, 10))  # & (df['mcap_d']>2)
        m = df.loc[ind,:].groupby(['evttime','news'])['abs_abret_norm'].mean().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
        s= df.loc[ind,:].groupby(['evttime','news'])['abs_abret_norm'].std().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
        c= df.loc[ind,:].groupby(['evttime','news'])['permno'].count().reset_index().pivot(columns='news',index='evttime',values='permno')
        plot_ev(m, s, c, do_cumulate=False, label_txt='Covered')
        plt.tight_layout()
        plt.savefig(save_dir+big+'.png')
        plt.tight_layout()
        plt.title(f'{big}')
        plt.tight_layout()
        plt.show()



    # df.groupby('big')['mcap_d'].unique()
    df['big']= df['mcap_d']
    sizes = np.sort(df['big'].unique())
    for big in sizes:
        if big ==10:
            plt.figure(figsize=[6.4*2,4.8])
            ind = (df['big'] == big) & (df['evttime'].between(-20, 20))  # & (df['mcap_d']>2)
        else:
            ind = (df['big'] == big) & (df['evttime'].between(-10, 10))  # & (df['mcap_d']>2)
        m = df.loc[ind,:].groupby(['evttime','news'])['abs_abret_norm'].mean().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
        s= df.loc[ind,:].groupby(['evttime','news'])['abs_abret_norm'].std().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
        c= df.loc[ind,:].groupby(['evttime','news'])['permno'].count().reset_index().pivot(columns='news',index='evttime',values='permno')
        plot_ev(m, s, c, do_cumulate=False, label_txt='Covered')
        plt.tight_layout()
        plt.savefig(save_dir+f'q{big}.png')
        plt.tight_layout()
        plt.title(f'{big}')
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=[6.4 * 2, 4.8])
    m = df.loc[:,:].groupby(['evttime','news'])['abs_abret_norm'].mean().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
    s= df.loc[:,:].groupby(['evttime','news'])['abs_abret_norm'].std().reset_index().pivot(columns='news',index='evttime',values='abs_abret_norm')
    c= df.loc[:,:].groupby(['evttime','news'])['permno'].count().reset_index().pivot(columns='news',index='evttime',values='permno')
    # plot_ev_no_conf(m, do_cumulate=False, label_txt='Covered')
    plot_ev(m, s, c, do_cumulate=False, label_txt='Covered')
    plt.savefig(save_dir+'intro_plot.png')
    plt.show()