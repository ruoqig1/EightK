import tqdm
import seaborn as sns
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
    nb_factors = 7
    winsorise_ret = -1
    winsorise_ret = -1
    do_cumulate = True
    # t_val = 1.96
    t_val = 2.58
    use_relase = False
    pred_col = 'pred'
    # pred_col = 'pred_rnd_2'
    use_ati = True
    # start_ret ='ret'
    start_ret ='abret'
    sigma_col = 'sigma_ret_train' if start_ret == 'ret' else 'sigma_ra'
    model_index = 1 # 2 is our main 1 is ok with atis...
    # for model_index in range(12):
    for model_index in [2]:
        if use_relase:
            save_dir = Constant.EMB_PAPER + 'release/'
        else:
            save_dir = Constant.EMB_PAPER + 'news/'
        if use_ati:
            save_dir = 'res/ati_plot_3/'
        os.makedirs(save_dir,exist_ok=True)
        if use_ati:
            print('load new')
            load_dir = 'res/model_tf_ati_3/'
            os.listdir(load_dir)
            df = pd.read_pickle(load_dir + f'new_{model_index}.p')
            par = Params()
            par.load(load_dir, f'/par_{model_index}.p')
        else:
            df, par = data.load_ml_forecast_draft_1()
        t=data.load_ati_cleaning_df()[['form_id','permno']].drop_duplicates()
        t['form_id'] = t['form_id'].apply(lambda x: x.replace('-',''))
        df = df.drop(columns='permno').merge(t)

        print(df.groupby(df['date'].dt.year)['permno'].count())
        if use_relase:
            df = df.merge(data.get_press_release_bool_per_event())
        else:
            df['release'] = 1

        df= df.groupby(['date','permno','news0','release'])['pred_prb','abret'].mean().reset_index()
        df['pred']  = np.sign(df['pred_prb']-0.5)
        # df['pred']  = np.sign(df['pred_prb']-df['pred_prb'].mean())
        # df['pred']  = np.sign(df['pred_prb']-df.groupby(df.date.dt.year)['pred_prb'].transform('mean'))

        if use_rav_cov_news_v2:
            rav = data.load_rav_coverage_split_by(False)
            df = df.merge(rav,how='left').fillna(0.0)
            df['news0'] = 1.0*(df['article']>0)

        if use_relase:
            df['news0'] = df['release']
        df['pred'] = df['pred'].replace({0:1})

        df['acc'] = df['pred']==np.sign(df['abret'])

        print(par.train.abny,par.train.l1_ratio, par.train.norm.name)

        # acc = df.groupby('items')['acc'].aggregate(['mean','count']).sort_values('mean')
        # item_to_keep = acc.loc[acc['count']>50,:].index

        # df = df.loc[~df['items'].isin([5.06,5.01,4.02]),:]
        # df = df.loc[df['items'].isin(item_to_keep),:]
        ev = data.load_abn_return(model=nb_factors,with_alpha=False)

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
        ind_time = (df['evttime']>=-3) & (df['evttime']<= 40)
        ind_time = (df['evttime']>=-1) & (df['evttime']<= 10)
        # ind_time = (df['evttime']>=-1) & (df['evttime']<= 60)


        size_ind = df['mcap_d']<=10
        # size_ind = df['mcap_d']>=5
        df['pred_rnd'] = np.sign(np.random.normal(size=df['pred'].shape))
        df['pred_rnd_2'] = np.sign(np.random.normal(size=df['pred'].shape)+0.5)


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))  # 2 subplots side by side
        n_list = [False,True] if use_relase else [0,1]
        k = 0
        for n in n_list:
            ind = df['news0']==n
            k+=1
            if n == -1:
                ind = df['news0']<=100
            m = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].mean().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
            s = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[sigma_col].mean().reset_index().pivot(columns=pred_col, index='evttime', values=sigma_col)
            c = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].count().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
            plt.subplot(2,2,k)
            plot_ev(m, s, c, do_cumulate=True, label_txt='Prediciton')
            plt.tight_layout()
            plt.title(f'{n}')
            plt.tight_layout()

        # LONG SHORT EV
        df['sign_ret'] = df[start_ret]*df[pred_col]

        m = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].mean().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
        s = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])[sigma_col].mean().reset_index().pivot(columns='news0', index='evttime', values=sigma_col)
        c = df.loc[ind_time & size_ind, :].groupby(['evttime', 'news0'])['sign_ret'].count().reset_index().pivot(columns='news0', index='evttime', values='sign_ret')
        k+=1
        plt.subplot(2, 2, k)
        plot_ev(m,s,c,do_cumulate = True,label_txt = 'PR' if use_relase else 'Covered')
        plt.tight_layout()
        plt.title('LONG SHORT EW')
        plt.tight_layout()


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
        k+=1
        plt.subplot(2, 2, k)
        plot_ev(m,s,c,do_cumulate = True,label_txt = 'PR' if use_relase else 'Covered')


        # par.train.abny,par.train.l1_ratio, par.train.norm.name
        n = f'abn{par.train.abny}l1{par.train.l1_ratio}norm{par.train.norm}'

        plt.title(n)
        plt.tight_layout()
        save_dir = save_dir+f'fact{nb_factors}/'
        os.makedirs(save_dir,exist_ok=True)
        save_dest  =save_dir+f'{model_index}_{n}.png'
        plt.savefig(save_dest)
        print('saved to ',save_dest)
        plt.show()


