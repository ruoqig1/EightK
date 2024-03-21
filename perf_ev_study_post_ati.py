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
    save_csv_for_attila = False
    use_reuters_news = False
    use_rav_cov_news_v2 = False
    winsorise_ret = -1
    news_columns = 'news_with_time' # 'news0_nr','news_with_time_nr','news_with_time'
    # news_columns = 'news_with_time'
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
    nb_factors = 6
    # for nb_factors in [6,1,2,3,7]:
    # for news_columns in ['news0', 'news0_nr', 'news_with_time', 'news_with_time_nr']:
    for news_columns in ['news0']:
        for model_index in [1]:
            for nb_factors in [66]:
                save_csv = Constant.DRAFT_1_CSV_PATH +f'MAIN_PLOT_modelid_{model_index}_factorid_{nb_factors}_newstype_{news_columns}'
                save_dir = Constant.EMB_PAPER + f'fig_exp/A/{news_columns.replace("_","")}/{model_index}/'
                os.makedirs(save_dir,exist_ok=True)

                # load the model prediciton
                load_dir = Constant.PATH_TO_MODELS_NOW
                os.listdir(load_dir)
                df = pd.read_pickle(load_dir + f'new_{model_index}.p')
                par = Params()
                par.load(load_dir, f'/par_{model_index}.p')
                par.print_values()

                t=data.load_ati_cleaning_df()[['form_id','permno']].drop_duplicates()
                t['form_id'] = t['form_id'].apply(lambda x: x.replace('-',''))
                filter_of_ati_to_drop_wrong_forms = data.load_icf_ati_filter(False, False)
                filter_of_ati_to_drop_wrong_forms=filter_of_ati_to_drop_wrong_forms[['date', 'permno']].drop_duplicates()
                df = df.drop(columns='permno').merge(t)
                df = df.merge(filter_of_ati_to_drop_wrong_forms)
                # change the news0
                df = df.drop(columns = 'news0')
                news = data.load_news0_post_ati_change()
                news['form_id'] = news['form_id'].apply(lambda x: x.replace('-',''))
                news['news0']*=1
                df = df.merge(news)

                model_factor = 6
                if model_factor != 66:
                    ind = pd.to_datetime(df['atime'], format='%H%M%S').dt.hour>16
                    df.loc[ind,'date']+=pd.DateOffset(days=1)
                    ind_friday = df['date'].dt.dayofweek == 4
                    df.loc[ind & ind_friday,'date']+=pd.DateOffset(days=2)




                df= df.groupby(['date','permno',news_columns])['pred_prb','abret'].mean().reset_index()
                df['pred']  = np.sign(df['pred_prb']-0.5)
                if save_csv_for_attila:
                    df[['date','permno','pred_prb','pred']].to_csv(Constant.DRAFT_1_CSV_PATH+f'prediction_of_model_{model_index}.csv', index=False)

                df['pred'] = df['pred'].replace({0:1})
                df['acc'] = df['pred']==np.sign(df['abret'])

                ev = data.load_abn_return(model=model_factor,with_alpha=False)
                # ev = data.load_abn_return(model=6,with_alpha=False)
                print(ev)
                ev.groupby('evttime')['abret'].mean().cumsum().plot()
                plt.show()


                ev['date'] = pd.to_datetime(ev['date'])
                ff = data.load_ff5()[['date','rf']]
                # ev = ev.merge(ff)
                # ev['abret'] += ev['rf']

                if winsorise_ret>0:
                    for model_index in tqdm.tqdm(ev['evttime'].unique(), 'winsorize'):
                        ind = ev['evttime'] == model_index
                        ev.loc[ind,'abret'] = PandasPlus.winzorize_series(ev.loc[ind,'abret'], winsorise_ret)
                        ev.loc[ind,'ret'] = PandasPlus.winzorize_series(ev.loc[ind,'ret'], winsorise_ret)

                df=df[['pred',news_columns,'date','permno']].merge(ev)
                df['year'] = df['date'].dt.year
                df = df.merge(data.load_mkt_cap_yearly())

                df['year'] = df['date'].dt.year

                df = df.dropna()

                df['pred'].mean()
                # ind_time = (df['evttime']>=-3) & (df['evttime']<= 15)
                # ind_time = (df['evttime']>=-10) & (df['evttime']<= 15)
                ind_time = (df['evttime']>=-10) & (df['evttime']<= 60)


                size_ind = df['mcap_d']<=10
                # size_ind = df['mcap_d']>=5
                df['pred_rnd'] = np.sign(np.random.normal(size=df['pred'].shape))
                df['pred_rnd_2'] = np.sign(np.random.normal(size=df['pred'].shape)+0.5)


                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))  # 2 subplots side by side
                n_list = [False,True] if use_relase else [0,1]
                k = 0
                for n in n_list:
                    ind = df[news_columns]==n
                    k+=1
                    if n == -1:
                        ind = df[news_columns]<=100
                    m = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].mean().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
                    s = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[sigma_col].mean().reset_index().pivot(columns=pred_col, index='evttime', values=sigma_col)
                    c = df.loc[ind_time & ind & size_ind, :].groupby(['evttime', pred_col])[start_ret].count().reset_index().pivot(columns=pred_col, index='evttime', values=start_ret)
                    plt.subplot(2,2,k)
                    print(n)
                    covered_txt = '_split_covered' if n ==1 else '_split_uncovered'
                    plot_ev(m, s, c, do_cumulate=True, label_txt='Prediciton', save_csv_path_and_name=save_csv+f'{covered_txt}.csv')
                    plt.tight_layout()
                    plt.title(f'{n}')
                    plt.tight_layout()

                # LONG SHORT EV
                df['sign_ret'] = df[start_ret]*df[pred_col]

                m = df.loc[ind_time & size_ind, :].groupby(['evttime', news_columns])['sign_ret'].mean().reset_index().pivot(columns=news_columns, index='evttime', values='sign_ret')
                s = df.loc[ind_time & size_ind, :].groupby(['evttime', news_columns])[sigma_col].mean().reset_index().pivot(columns=news_columns, index='evttime', values=sigma_col)
                c = df.loc[ind_time & size_ind, :].groupby(['evttime', news_columns])['sign_ret'].count().reset_index().pivot(columns=news_columns, index='evttime', values='sign_ret')
                k+=1
                plt.subplot(2, 2, k)
                plot_ev(m,s,c,do_cumulate = True,label_txt = 'PR' if use_relase else 'Covered', save_csv_path_and_name=save_csv+'_long_short_equally_weighted.csv')
                plt.tight_layout()
                plt.title('LONG SHORT EW')
                plt.tight_layout()


                size_ind = df['mcap_d']<=8
                df['w_ret2'] = df[start_ret]*df['mcap']
                long_short = []
                for pred in [-1,1]:
                    m = df.loc[ind_time & size_ind & (df[pred_col]==pred), :].groupby(['evttime', news_columns])['w_ret2'].sum().reset_index().pivot(columns=news_columns, index='evttime', values='w_ret2')
                    norm = df.loc[ind_time & size_ind & (df[pred_col]==pred), :].groupby(['evttime', news_columns])['mcap'].sum().reset_index().pivot(columns=news_columns, index='evttime', values='mcap')
                    m = m/norm
                    long_short.append(m)
                m = long_short[1]-long_short[0]
                s = df.loc[ind_time & size_ind, :].groupby(['evttime', news_columns])[sigma_col].mean().reset_index().pivot(columns=news_columns, index='evttime', values=sigma_col)
                c = df.loc[ind_time & size_ind, :].groupby(['evttime', news_columns])['w_ret2'].count().reset_index().pivot(columns=news_columns, index='evttime', values='w_ret2')
                k+=1
                plt.subplot(2, 2, k)
                plot_ev(m, s, c, do_cumulate = True, label_txt = 'PR' if use_relase else 'Covered', save_csv_path_and_name=save_csv+'_long_short_value_weighted.csv')


                # par.train.abny,par.train.l1_ratio, par.train.norm.name
                plt.title(n)
                plt.tight_layout()
                os.makedirs(save_dir,exist_ok=True)
                save_dest  =save_dir+f'fact{nb_factors}.png'
                plt.savefig(save_dest)
                print('saved to ',save_dest)
                plt.show()
            # for model_index in range(12):


