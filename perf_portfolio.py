import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
from didipack.trainer.train_splitter import get_start_dates,get_chunks,get_tasks_for_current_node
import psutil
from train_main import set_ids_to_eight_k_df
from experiments_params import get_main_experiments
from scipy import stats
from matplotlib import pyplot as plt




if __name__ == "__main__":
    args = didi.parse()
    par = Params()
    data = Data(par)
    i_range =[2,4,5,6,7]
    load_dir ='res/model_daily_1/'

        # model_index = i_range[1]
    for model_index in i_range:
        par.load(load_dir,f'/par_{model_index}.p')
        df = pd.read_pickle(load_dir+f'/new_{model_index}.p').rename(columns={'ticker':'permno'})
        crsp = data.load_crsp_daily()
        crsp['ret'] = pd.to_numeric(crsp['ret'],errors='coerce')
        df=df.merge(crsp[['date','ticker','permno','ret']])

        df =df.groupby(['permno','date'])[['ret','y_pred_prb','y_true']].mean().reset_index()

        df['year'] = df['date'].dt.year
        print(df.groupby('year')['y_pred_prb'].mean())
        df['tresh'] = df.groupby('year')['y_pred_prb'].transform('mean')

        df['pred'] = df['y_pred_prb']>df['tresh']
        # df['pred'] = df['y_pred_prb']>0
        df['accuracy']=df['pred']==df['y_true']
        df['accuracy'].mean()

        df.groupby('date')['permno'].count().plot()
        df['pct']=df.groupby('date')['y_pred_prb'].rank(pct=True)

        tresh = 0.2
        df['pos'] = 1*(df['pct']>(1-tresh)) - (df['pct']<=tresh)*1
        ret=df.groupby(['date','pos'])['ret'].mean().reset_index().pivot(columns='pos',index='date',values='ret')
        ret[0] = ret[1]-ret[-1]
        ret.cumsum().plot()
        sh = np.sqrt(252)*(ret.mean()/ret.std()).round(3)
        plt.title(f'model_index {model_index}sharpe: {np.round(sh[0],3)}')
        plt.show()