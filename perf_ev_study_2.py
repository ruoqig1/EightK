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
    nb_groups = 3
    for nb_factors in [7]:
    # for nb_factors in [7]:
        winsorise_ret = -1
        do_cumulate = True
        t_val = 1.96
        # t_val = 2.58
        use_relase = False
        pred_col = 'pred'
        # pred_col = 'pred_rnd_2'
        use_ati = True
        # start_ret ='ret'
        start_ret ='abret'
        sigma_col = 'sigma_ret_train' if start_ret == 'ret' else 'sigma_ra'
        model_index = 1 # 2 is our main 1 is ok with atis...

        save_dir = Constant.EMB_PAPER + 'fig_exp/B/'
        for model_index in [1]:
        # for model_index in range(12):
            if use_ati:
                print('load new')
                load_dir = Constant.PATH_TO_MODELS_NOW
                os.listdir(load_dir)
                df = pd.read_pickle(load_dir + f'new_{model_index}.p')
                par = Params()
                par.load(load_dir, f'/par_{model_index}.p')
            else:
                df, par = data.load_ml_forecast_draft_1()
            t=data.load_ati_cleaning_df()[['form_id','permno']].drop_duplicates()
            t['form_id'] = t['form_id'].apply(lambda x: x.replace('-',''))
            df = df.drop(columns='permno').merge(t)

            # print(df.groupby(df['date'].dt.year)['permno'].count())
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

            ind_time = (df['evttime']>=-5) & (df['evttime']<= 40)
            ind_time = (df['evttime']>=-3) & (df['evttime']<= 15)

            size_ind = df['mcap_d']<=10
            if nb_groups == 10:
                size_ind_list = [df['mcap_d']<=10] + [df['mcap_d']==(i+1) for i in range(10)]
            elif nb_groups == 3:
                size_ind_list = [df['mcap_d']<=10] + [df['mcap_d']==(i+1) for i in range(10)]
            else:
                size_ind_list = [df['mcap_d']<=10] + [df['mcap_d'].between((i*2)-1,i*2) for i in range(1,6)]

            # size_ind = df['mcap_d']>=5
            df['pred_rnd'] = np.sign(np.random.normal(size=df['pred'].shape))
            df['pred_rnd_2'] = np.sign(np.random.normal(size=df['pred'].shape)+0.5)


            nb_row = nb_groups+1
            fig, axes = plt.subplots(nrows=nb_row, ncols=2, figsize=(12, 6*nb_row))  # 2 subplots side by side
            n_list = [False,True] if use_relase else [0,1]
            k = 0
            for dec, size_ind in tqdm.tqdm(enumerate(size_ind_list),f'Model {model_index}'):
                for n in n_list:
                    ind = df['news0']==n
                    k+=1
                    if n == -1:
                        ind = df['news0']<=100
                    m = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].mean().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
                    s = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[sigma_col].mean().reset_index().pivot(columns=pred_col, index='evttime', values=sigma_col)
                    c = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].count().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
                    # m_all = df.loc[ind_time & ind & size_ind, :].groupby(['evttime'])[start_ret].mean()
                    # s_all = df.loc[ind_time & ind & size_ind, :].groupby(['evttime'])[sigma_col].mean()
                    # c_all = c.sum(1)
                    # m['all'] = m_all
                    # s['all'] =s_all
                    # c['all'] =c_all
                    mean_mcap = np.round(df.loc[(size_ind & ind_time),'mcap'].mean()/1e6,3)
                    plt.subplot(nb_row, 2, k)
                    plot_ev(m, s, c, do_cumulate=True, label_txt='Prediciton')
                    plt.title(f'{n} Quantile={dec} (mcap={mean_mcap})')
            plt.tight_layout()
            final_dir = save_dir+f'fact{nb_factors}/'
            os.makedirs(final_dir,exist_ok=True)
            save_dest  =final_dir+f'{model_index}.png'
            plt.savefig(save_dest)
            plt.show()

