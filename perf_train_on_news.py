import numpy as np
import pandas as pd
import os

import tqdm

from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates, get_chunks, get_tasks_for_current_node
import psutil
from utils_local.general import *
from utils_local.trainer_specials import *
from experiments_params import get_main_experiments
import tensorflow as tf
from utils_local.plot import plot_ev
from matplotlib import pyplot as plt


if __name__ == '__main__':
    par = get_main_experiments(0,train=False)
    data = Data(par)
    n = pd.read_pickle('data/p/news_per_stock_id.p')[['id','alert','timestamp','atime']]

    for index_model in range(2):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for alert in [False, True]:
            df = pd.read_pickle(f'res/model_tf_4/new_{index_model}.p')
            par.load(f'res/model_tf_4/',f'/par_{index_model}.p')
            print('MODEL,',par.train.l1_ratio,par.train.abny)
            str_title = (f'Reuters {par.train.filter_on_reuters}, Alert {par.train.filter_on_alert}, Cosine {par.train.filter_on_cosine}, Prn {par.train.filter_on_prn}, Abny {par.train.abny}')
            # str_title = f'L1_ratio {par.train.l1_ratio}, Abny {par.train.abny}'

            df =df.merge(n)

            df['time_news'] = df['timestamp'].apply(lambda x: str(x).split('T')[1].split('.')[0])
            df = df.loc[df['alert']==alert,:]

            # Convert string columns to datetime.time format
            df['time_news'] = pd.to_datetime(df['time_news']).dt.time
            # df['atime'] = pd.to_datetime(df['atime']).dt.time
            # Check if 'atime' is after 'time_news'
            # df = df.loc[df['atime'] < df['time_news'],:]



            df = df.rename(columns={'ticker':'permno'})
            df['year'] = df['date'].dt.year
            df['y_pred']=(df['y_pred_prb']>df.groupby('year')['y_pred_prb'].transform('mean'))*1
            df['accuracy'] = df['y_pred']==df['y_true']
            print(df['accuracy'].mean())
            df = df.groupby(['date','permno'])[['y_pred','y_true','y_pred_prb']].mean().reset_index()
            df['year'] = df['date'].dt.year
            df['y_pred']=(df['y_pred_prb']>df.groupby('year')['y_pred_prb'].transform('mean'))*1


            ev = pd.read_pickle(data.p_dir + 'abn_ev_monly.p')

            df =df.merge(ev)

            ind_time = df['evttime'].between(-1,20)
            ind = df['evttime'].between(-20,20)

            df = df.merge(data.load_mkt_cap_yearly())
            # ind = df['mcap_d']==5
            group_col = 'y_pred'
            start_ret ='abret'

            m = df.loc[ind_time & ind, :].groupby(['evttime', group_col])[start_ret].mean().reset_index().pivot(columns=group_col, index='evttime', values=start_ret)
            s = df.loc[ind_time & ind, :].groupby(['evttime', group_col])['sigma_ra'].mean().reset_index().pivot(columns=group_col, index='evttime', values='sigma_ra')
            c = df.loc[ind_time & ind, :].groupby(['evttime', group_col])[start_ret].count().reset_index().pivot(columns=group_col, index='evttime', values=start_ret)
            plot_ev(m, s, c, do_cumulate=True, label_txt='Pred', ax=axes[int(alert*1)],title=f'{str_title} \n Alert {alert}')
            # plt.title(str_title)
        fig.suptitle(f'{str_title}', fontsize=16, y=1.05)
        fig.tight_layout()
        plt.show()

