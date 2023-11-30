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
    save_attilah = False
    nb_factors = 7

    save_dir = Constant.EMB_PAPER+'ss/'
    csv_dir = Constant.DRAFT_1_CSV_PATH
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)


    # load and process all the ravenpack news
    rav = data.load_ravenpack_all()
    rav['pr_r'] = rav['news_type']=='PRESS-RELEASE'
    rav['pr_r'] = rav['pr_r'].map({True:'press',False:'article'}) # press=press_relase, article=news_article
    temp =rav.groupby(['rdate','permno','pr_r'])['relevance'].max().reset_index()
    temp = temp.pivot(columns='pr_r',index=['rdate','permno'],values='relevance').fillna(0.0).reset_index()
    temp['permno'] = temp['permno'].astype(int)
    temp = temp.rename(columns={'rdate':'date'})

    # load and process all the metadata for 8k
    df = data.load_some_relevance_icf()
    df=df.rename(columns={'adate':'date'})
    # merge wit ravenpack
    df = df.merge(temp,how='left')
    df['article'] = df['article'].fillna(0.0)
    df['npress2'] = df['press'].fillna(0.0)
    # now news0 is the old definition (press +article), press is the press release, article is the article in ravenpack
    df['n2'] = df['article'].fillna(0.0)+df['press'].fillna(0.0)
    df['news'] = (df['n2'].fillna(0.0)>0)

    # define big cap as firm in mc_ap decile
    df['big'] = df['mcap_d']==10
    # drop from the sample the items not in the correct list
    df = df.loc[df['items'].isin(Constant.LIST_ITEMS_TO_USE),:]

    # load and merge with the abnormal returns.
    ev = data.load_abn_return(model=nb_factors)
    df = df.merge(ev)


    # add the snp 500 boolean
    sp = data.load_snp_const(False)
    sp['ym'] = PandasPlus.get_ym(sp['date'])
    sp = sp.drop(columns='date')
    df['ym'] = PandasPlus.get_ym(df['date'])
    df = df.merge(sp, how='left')

    df['in_snp'] = (((df['date'] <= df['ending']) & (df['date'] >= df['start'])) * 1).fillna(0.0)

    df = df[['date','permno','abs_abret','sigma_abs_ra','sigma_ra','news','big','mcap_d','evttime']].drop_duplicates()
    # define the big micro mega cap a bit arbitrairly based on decile
    df['big'] = 'micro'
    df.loc[df['mcap_d']<=10,'big'] = 'Mega'
    df.loc[df['mcap_d']<10,'big'] = 'Large'
    df.loc[df['mcap_d']<6,'big'] = 'Small'
    df.loc[df['mcap_d']<3,'big'] = 'Micro'
    sizes = ['Mega','Large','Small','Micro']


    # df['big'] = df['in_snp'].replace({1:'snp',0:'non_snp'})
    # sizes =df['big'].unique()

    t = df.groupby(['date', 'permno'])['abs_abret'].transform('mean')
    df['abs_abret_norm'] = df['abs_abret'] - t


    # main analysis by arbitrarly defined mega,large,small,micro.
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



    # running the main analysis by mcap_d
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