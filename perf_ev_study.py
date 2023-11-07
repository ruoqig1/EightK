import tqdm

from data import Data
from utils_local.general import *
from matplotlib import pyplot as plt

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from didipack import PandasPlus
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils_local.plot import plot_ev


if __name__ == "__main__":

    par = Params()
    data = Data(par)
    use_reuters_news = False
    use_rav_cov_news_v2 = False
    use_constant_in_abret = False
    winsorise_ret = -1
    do_cumulate = True
    # t_val = 1.96
    t_val = 2.58
    use_relase = False
    pred_col = 'pred'
    # pred_col = 'pred_rnd_2'
    model_index = 2 # 2 is our main
    start_ret ='abret'
    sigma_col = 'sigma_ret_train' if start_ret == 'ret' else 'sigma_ra'

    if use_relase:
        save_dir = Constant.EMB_PAPER + 'release/'
    else:
        save_dir = Constant.EMB_PAPER + 'news/'
    os.makedirs(save_dir,exist_ok=True)
    # load models
    # load_dir = 'res/temp_new/'
    load_dir = 'res/model_final_res/'
    # load_dir = 'res/model_final_long/'
    os.listdir(load_dir)
    df = pd.read_pickle(load_dir + f'new_{model_index}.p')
    print(df.groupby(df['date'].dt.year)['permno'].count())
    if use_relase:
        df = df.merge(data.get_press_release_bool_per_event())
    else:
        df['release'] = 1

    df= df.groupby(['date','permno','news0','release'])['pred_prb','abret'].mean().reset_index()
    # df['pred']  = np.sign(df['pred_prb']-0.5)
    df['pred']  = np.sign(df['pred_prb']-df['pred_prb'].mean())

    if use_reuters_news:
        news = pd.read_pickle('data/cleaned/some_news/ref.p')
        news = news.loc[news['alert']==False,:]
        news['date'] = pd.to_datetime(news['timestamp'].apply(lambda x: x.split('T')[0]))
        news['news0'] = True
        news = news.merge(data.load_crsp_daily()[['permno','ticker','date']])
        df = df.drop(columns='news0').merge(news[['date','permno','news0']],how='left')
        df['news0'] = df['news0'].fillna(value = False)

    if use_rav_cov_news_v2:
        rav = data.load_rav_coverage(False)
        df = df.merge(rav,how='left').fillna(0.0)
        df['news0'] = 1.0*(df['article']>0)

    if use_relase:
        df['news0'] = df['release']
    df['pred'] = df['pred'].replace({0:1})

    df['acc'] = df['pred']==np.sign(df['abret'])


    print(df.shape)
    par.load(load_dir,f'/par_{model_index}.p')
    print(par.train.abny,par.train.l1_ratio)
    # acc = df.groupby('items')['acc'].aggregate(['mean','count']).sort_values('mean')
    # item_to_keep = acc.loc[acc['count']>50,:].index

    # df = df.loc[~df['items'].isin([5.06,5.01,4.02]),:]
    # df = df.loc[df['items'].isin(item_to_keep),:]

    if use_constant_in_abret:
        ev = pd.read_pickle(data.p_dir+'abn_ev_m.p')
    else:
        ev = pd.read_pickle(data.p_dir+'abn_ev_monly.p')

    if winsorise_ret>0:
        for model_index in tqdm.tqdm(ev['evttime'].unique(), 'winsorize'):
            ind = ev['evttime'] == model_index
            ev.loc[ind,'abret'] = PandasPlus.winzorize_series(ev.loc[ind,'abret'], winsorise_ret)
            ev.loc[ind,'ret'] = PandasPlus.winzorize_series(ev.loc[ind,'ret'], winsorise_ret)


    df=df[['pred','news0','date','permno']].merge(ev)
    df['year'] = df['date'].dt.year
    df = df.merge(data.load_mkt_cap_yearly())

    df['good_eight_k'] = df['abret'] > 0
    df['year'] = df['date'].dt.year

    # df.groupby(['permno', 'mcap_d'])['good_eight_k'].mean().reset_index().groupby('mcap_d')['good_eight_k'].mean().plot()
    # plt.show()

    df = df.dropna()

    df['pred'].mean()
    ind_time = (df['evttime']>=-3) & (df['evttime']<= 20)
    ind_time = (df['evttime']>=-5) & (df['evttime']<= 20)



    size_ind = df['mcap_d']<=10
    df['pred_rnd'] = np.sign(np.random.normal(size=df['pred'].shape))
    df['pred_rnd_2'] = np.sign(np.random.normal(size=df['pred'].shape)+0.5)

    n_list = [False,True] if use_relase else [0,1]
    for n in n_list:
        ind = df['news0']==n
        if n == -1:
            ind = df['news0']<=100
        m = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].mean().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
        s = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[sigma_col].mean().reset_index().pivot(columns=pred_col, index='evttime', values=sigma_col)
        c = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].count().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
        plot_ev(m, s, c, do_cumulate=True, label_txt='Prediciton')
        plt.tight_layout()
        plt.savefig(save_dir+f'car_n{n}.png')
        plt.show()

    # LONG SHORT EV
    df['sign_ret'] = df[start_ret]*df[pred_col]

    m = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].mean().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
    s = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])[sigma_col].mean().reset_index().pivot(columns='news0', index='evttime', values=sigma_col)
    c = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].count().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
    plot_ev(m,s,c,do_cumulate = True,label_txt = 'PR' if use_relase else 'News')
    plt.tight_layout()
    plt.savefig(save_dir+f'ls_ew.png')
    plt.title('LONG SHORT EW')
    plt.tight_layout()
    plt.show()


    # ev = data.load_some_relevance_icf()[['adate','permno','items']].rename(columns={'adate':'date'})
    # temp = df.merge(ev)
    #
    # for items in Constant.LIST_ITEMS_TO_USE:
    #     ind_items = temp['items']==items
    #     m = temp.loc[ind_items & ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].mean().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
    #     s = temp.loc[ind_items & ind_time & size_ind, :].groupby(['evttime', 'news0'])[sigma_col].mean().reset_index().pivot(columns='news0', index='evttime', values=sigma_col)
    #     c = temp.loc[ind_items & ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].count().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
    #     plot_ev(m, s, c, do_cumulate=True, label_txt='PR' if use_relase else 'News')
    #     plt.tight_layout()
    #
    #     plt.title(f'LONG SHORT EW {items}')
    #     plt.tight_layout()
    #     plt.show()
    #
    # breakpoint()

    # #LONG SHORT VW CLEAN
    print(df.shape)

    size_ind = df['mcap_d']<=8
    df['w_ret2'] = df[start_ret]*df['mcap']
    long_short = []
    for pred in [-1,1]:
        m = df.loc[ind_time & size_ind & (df[pred_col]==pred), :].groupby(['evttime', 'news0'])['w_ret2'].sum().reset_index().pivot(columns='news0', index='evttime', values='w_ret2')
        norm = df.loc[ind_time & size_ind & (df[pred_col]==pred), :].groupby(['evttime', 'news0'])['mcap'].sum().reset_index().pivot(columns='news0', index='evttime', values='mcap')
        m = m/norm
        long_short.append(m)
    m = long_short[1]-long_short[0]
    s = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])[sigma_col].mean().reset_index().pivot(columns='news0', index='evttime', values=sigma_col)
    c = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])['w_ret2'].count().reset_index().pivot(columns='news0', index='evttime', values='w_ret2')
    plot_ev(m,s,c,do_cumulate = True,label_txt = 'PR' if use_relase else 'News')
    plt.tight_layout()
    plt.savefig(save_dir+f'ls_vw.png')
    plt.title('LONG SHORT VW')
    plt.tight_layout()
    plt.show()



    ###### Additional plots with no predicitons.




    ind_time = (df['evttime']>=-20) & (df['evttime']<= 20)
    df

    # df.loc[:, 'abs_abret'] = PandasPlus.winzorize_series(df.loc[:, 'abs_abret'], 1)
    # df['big'] = df['mcap_d']==10
    # group_col = 'news0'
    # for n in [True,False]:
    #     ind = df['big']==n
    #     m = df.loc[ind_time & ind, :].groupby(['evttime', group_col])['abs_abret'].mean().reset_index().pivot(columns=group_col, index='evttime', values='abs_abret')
    #     s = df.loc[ind_time & ind, :].groupby(['evttime', group_col])['sigma_abs_ra'].mean().reset_index().pivot(columns=group_col, index='evttime', values='sigma_abs_ra')
    #     c = df.loc[ind_time & ind, :].groupby(['evttime', group_col])[start_ret].count().reset_index().pivot(columns=group_col, index='evttime', values=start_ret)
    #     plot_ev(m, s, c, do_cumulate=False, label_txt='News')
    #     plt.tight_layout()
    #     plt.savefig(save_dir+f'news_abret.png')
    #     plt.title(f'Big Firm {n}')
    #     plt.tight_layout()
    #     plt.show()
