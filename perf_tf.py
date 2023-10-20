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

def load_res_tf(par:Params):
    temp_save_dir = par.get_res_dir()
    # to uncoment with the line commented a bit above for the one ok ish result
    res = pd.DataFrame()
    for f in np.sort(os.listdir(temp_save_dir)):
        t = pd.read_pickle(temp_save_dir + f)
        res = pd.concat([res, t], axis=0)
    res['date']=pd.to_datetime(res['date'].apply(lambda x: x.split(' ')[0]))
    return res


if __name__ == "__main__":
    args = didi.parse()
    par = Params()
    ret_cols = ['ret_m', 'ret_5', 'ret_20']

    temp_dir ='res/model_tf_2/'
    os.makedirs(temp_dir,exist_ok=True)
    for i in range(6):
    # for i in [0]:
        par = get_main_experiments(i,train=False)
        try:
            df = load_res_tf(par)
            df.to_pickle(f'{temp_dir}/new_{i}.p')
            par.save(save_dir=temp_dir, file_name=f'par_{i}.p')
            print(df['date'].dt.year.unique())
            # print('ACCURACY OVERALL',df['acc'].mean())
            # print('Sanity check',df['pred'].mean())
        except:
            print(i,'bugged')

