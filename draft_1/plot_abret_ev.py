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


def print_size(df,step):
    print(f'### Step {step}')
    print(df.groupby('alert')['body'].count()/1e6,flush=True)

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    use_constant_in_abret = False
    save_dir = Constant.EMB_PAPER+'ss/'
    os.makedirs(save_dir, exist_ok=True)


    rav = data.load_ravenpack_all()


    df = data.load_some_relevance_icf()
    df['news'] = df['no_rel']==0
    df=df.rename(columns={'adate':'date'})
    # df = df.loc[df['ddate'].dt.year>=2012,:]
    df['big'] = df['mcap_d']==10
    df = df.loc[df['items'].isin(Constant.LIST_ITEMS_TO_USE),:]

    if use_constant_in_abret:
        ev = pd.read_pickle(data.p_dir+'abn_ev_m.p')
    else:
        ev = pd.read_pickle(data.p_dir+'abn_ev_monly.p')

    df = df.merge(ev)
    df = df[['date','permno','abs_abret','sigma_abs_ra','sigma_ra','news','big','mcap_d','evttime']].drop_duplicates()
    df
    df['sigma_to_avg'] = df['sigma_abs_ra']**2
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

    t = df.groupby(['date', 'permno'])['abs_abret'].transform('mean')
    df['abs_abret_norm'] = df['abs_abret'] - t


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
        # s= df.loc[ind,:].groupby(['evttime','news'])['sigma_to_avg'].mean().reset_index().pivot(columns='news',index='evttime',values='sigma_to_avg')**(1/2)
        c= df.loc[ind,:].groupby(['evttime','news'])['permno'].count().reset_index().pivot(columns='news',index='evttime',values='permno')
        plot_ev(m, s, c, do_cumulate=False, label_txt='News')
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
        # s= df.loc[ind,:].groupby(['evttime','news'])['sigma_to_avg'].mean().reset_index().pivot(columns='news',index='evttime',values='sigma_to_avg')**(1/2)
        c= df.loc[ind,:].groupby(['evttime','news'])['permno'].count().reset_index().pivot(columns='news',index='evttime',values='permno')
        plot_ev(m, s, c, do_cumulate=False, label_txt='News')
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
    plot_ev_no_conf(m, do_cumulate=False, label_txt='News')
    plt.savefig(save_dir+'intro_plot.png')
    # plot_ev(m, s, c, do_cumulate=False, label_txt='News')
    plt.show()