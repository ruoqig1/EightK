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
    par = get_main_experiments(5,train=False)
    data = Data(par)
    n = pd.read_pickle('data/p/news_per_stock_id.p')[['id','alert','timestamp','atime']]

    for index_model in range(1):
        df = pd.read_pickle(f'res/model_tf_2/new_{index_model}.p')
        par.load(f'res/model_tf_2/',f'/par_{index_model}.p')
        print('MODEL,',par.train.l1_ratio,par.train.abny)
        str_title = f'L1_ratio {par.train.l1_ratio}, Abny {par.train.abny}'

        df =df.merge(n)

        df['time_news'] = df['timestamp'].apply(lambda x: str(x).split('T')[1].split('.')[0])
        df = df.loc[df['alert']==False,:]

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
        plot_ev(m, s, c, do_cumulate=True, label_txt='Pred')
        # plt.title(str_title)
        plt.tight_layout()
        plt.show()


        # t=df.groupby(['year'])['y_pred_prb'].rank(pct=True)
        #
        # df['buy'] =t>0.9
        # df['sell'] =t<0.1
        # df['pos'] = 1*(t>0.9) -1*(t<0.1)
        # start_ret ='ret'
        # df.groupby(['year','pos'])[start_ret].mean().reset_index().pivot(columns='pos',index='year',values=start_ret).plot()
        # plt.show()