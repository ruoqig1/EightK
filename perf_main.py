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
    breakpoint()
    res['pred_prb'] = res['pred']
    # res['pred'] = np.sign(res['pred']-0.5)
    df['id'] = df.index

    df = res.reset_index()[['t_ind', 'id', 'pred','pred_prb']].merge(df)
    df['mse'] = (df['pred'] - df['ret_m']) ** 2
    df['acc'] = np.sign(df['pred']) == np.sign(df['ret_m'])
    return df
# /data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/T_train360T_val36testing_window1shrinkage_list50pred_modelPredModel.RIDGEnormNormalisation.ZSCOREsave_insFalsetnews_onlyFalsel1_ratio0.5/OPT_13b/EIGHT_LEGAL/'
# /data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/T_train8T_val2testing_window1shrinkage_list4pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio0.0abnyTruemin_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/2013.p
if __name__ == "__main__":
    args = didi.parse()
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    df = load_res(par)
    print(par.train.T_train, par.train.l1_ratio, df.shape, par.train.tnews_only)
    print('Model:')
    print('ACCURACY OVERALL', df['acc'].mean())

    save_dir ='res/new_temp/'
    os.makedirs(save_dir,exist_ok=True)
    for i in range(3):
        par = get_main_experiments(i,train=False)
        breakpoint()
        try:
            df = load_res(par)
            print(par.train.T_train,par.train.l1_ratio,df.shape,par.train.tnews_only)
            print('Model:')
            print('ACCURACY OVERALL',df['acc'].mean())
            print('Sanity check',df['pred'].mean())
            df.to_pickle(save_dir+f'df_{i}.p')
            par.save(save_dir,f'params_{i}.p')
            print('saved')
        except:
            print(i,'bugged')


