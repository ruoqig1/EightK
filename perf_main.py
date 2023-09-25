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

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # memory in GB

def load_res(par:Params):
    load_dir = par.get_training_dir()
    df = pd.read_pickle(load_dir + 'main_df.p')
    # par.train.tnews_only = None
    df = set_ids_to_eight_k_df(df, par)
    df = df.reset_index(drop=True)
    df = df.sort_values('date')
    temp_save_dir = par.get_res_dir()
    # to uncoment with the line commented a bit above for the one ok ish result
    # temp_save_dir = '/data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list7pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsenb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/'
    res = pd.DataFrame()
    for f in np.sort(os.listdir(temp_save_dir)):
        t = pd.read_pickle(temp_save_dir + f)
        res = pd.concat([res, t], axis=0)
    df = res.reset_index()[['t_ind', 'id', 'pred']].merge(df)
    df['mse'] = (df['pred'] - df['ret_m']) ** 2
    df['acc'] = np.sign(df['pred']) == np.sign(df['ret_m'])
    return df

if __name__ == "__main__":
    args = didi.parse()

    for i in range(1):
        par = get_main_experiments(i,train=False)
        df = load_res(par)
        print(par.train.norm,par.train.l1_ratio,df.shape,par.train.tnews_only)
        print('Model:')
        print('ACCURACY OVERALL',df['acc'].mean())
        print('Sanity check',df['pred'].mean())


    grp = df['items']
    grp = df['date'].dt.year


    print('ACCURACY PER YEAR')
    print(df.groupby(grp)['acc'].aggregate(['mean','count']).sort_index())
    #
    # df = df.loc[~df['items'].isin([5.06, 4.02, 2.04, 5.01, 1.04, 2.05, 4.01, 5.08]),:]
    # df = df.loc[df['items'].isin([8.01]),:]

    ret_cols =['ret_m','ret_5','ret_20','ret_60','ret_250']
    print(df.groupby(['news0','pred'])[ret_cols].mean())


    # Grouping the DataFrame by 'news0'


    # Loop through each groupcd /sc
    for col in ret_cols:
        grouped = df.dropna(subset=col).groupby("news0")
        # Storing results
        results = {}
        for name, group in grouped:
            data_pred1 = group[group['pred'] == 1][col]
            data_pred_neg1 = group[group['pred'] == -1][col]
            # Assuming equal variance for now
            t_stat, p_val = stats.ttest_ind(data_pred1, data_pred_neg1, equal_var=False)
            results[name] = {'t_statistic': t_stat, 'p_value': p_val, 'mean_pred_pos':data_pred1.mean(), 'mean_pred_neg':data_pred_neg1.mean()}
        print('\n \n COL',col)
        print(pd.DataFrame(results).round(5))
