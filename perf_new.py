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
from experiments_params import *
from scipy import stats

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # memory in GB

def load_res(par:Params):
    load_dir = par.get_training_dir()
    df = pd.read_pickle(load_dir + 'main_df.p')
    df = set_ids_to_eight_k_df(df, par)
    df = df.reset_index(drop=True)
    df = df.sort_values('date')
    temp_save_dir = par.get_res_dir()
    # to uncoment with the line commented a bit above for the one ok ish result
    res = pd.DataFrame()
    print('start loading from',temp_save_dir)
    for f in np.sort(os.listdir(temp_save_dir)):
        t = pd.read_pickle(temp_save_dir + f)
        res = pd.concat([res, t], axis=0)
    res['pred_prb'] = res['pred']
    # res['pred'] = np.sign(res['pred']-0.5)
    res['pred'] = np.sign(res['pred']-res['pred'].mean())
    res['pred'] = res['pred'].replace({0.0:1.0})
    df = res.reset_index()[['id', 'pred','pred_prb']].merge(df)
    return df


if __name__ == "__main__":
    args = didi.parse()
    par = Params()
    ret_cols = ['ret_m', 'ret_5', 'ret_20']

    temp_dir ='res/temp_new/'
    temp_dir ='res/model_final_long/'
    temp_dir ='res/model_tf/'
    temp_dir ='res/model_tf_ati_2/'
    temp_dir ='res/model_tf_ati_news_var/'
    temp_dir ='res/model_tf_ati_dec/'
    temp_dir ='res/model_tf_ati_dec_spartan/'
    os.makedirs(temp_dir,exist_ok=True)
    for i in range(12):
    # for i in [0]:
        par = get_main_experiments_train_all(i,train=False)
        try:
            df = load_res(par)
            df.to_pickle(f'{temp_dir}/new_{i}.p')
            par.save(save_dir=temp_dir, file_name=f'par_{i}.p')
            print(df['date'].dt.year.unique())
            print('v1')
            df['acc'] = np.sign(df['ret'])==df['pred']
            df['acc_ab'] = np.sign(df['abret'])==df['pred']
            print(df[['acc','acc_ab']].mean())
            print('v2')
            df['pred'] = ((df['pred_prb']>df.groupby(df['date'].dt.year)['pred_prb'].transform('mean'))*2)-1
            df['acc'] = np.sign(df['ret'])==df['pred']
            df['acc_ab'] = np.sign(df['abret'])==df['pred']
            print(df[['acc','acc_ab']].mean())

            # print('Sanity check',df['pred'].mean())
        except:
            print(i,'bugged')

