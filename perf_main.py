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
def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # memory in GB

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL


    load_dir = par.get_training_dir()
    df= pd.read_pickle(load_dir+'main_df.p').reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    df =df.reset_index(drop=True)
    df=df.sort_values('date')

    par.train.testing_window = 1
    par.train.T_val = 6
    par.train.T_train =-60
    par.train.min_nb_chunks_in_cluster=5
    par.train.nb_chunks=20
    trainer = TrainerRidge(par)
    nb_nodes_total = 10
    verbose = True
    temp_save_dir = par.get_res_dir()
    res = pd.DataFrame()
    for f in tqdm.tqdm(np.sort(os.listdir(temp_save_dir)),'merge'):
        t = pd.read_pickle(temp_save_dir+f)
        res = pd.concat([res,t],axis=0)
    # res = res.reset_index().rename(columns={'level_1':'id'})
    # res['cik'] = res['id'].apply(lambda x: x.split('-')[0]).astype(int)
    # res['items'] = res['id'].apply(lambda x: x.split('-')[-1])
    # res['form_id'] = res['id'].apply(lambda x: x.split('-')[1])
    #
    # res[['cik','form_id','items']].drop_duplicates().shape

    1-((res['ret_m']-res['pred'])**2).sum()/((res['ret_m']-res['ret_m'].mean())**2).sum()
    1-((res['ret_m']-res['pred'])**2).sum()/((res['ret_m']-0)**2).sum()

    df=res.reset_index()[['t_ind','id','pred']].merge(df)
    df['mse']=(df['pred']-df['ret_m'])**2
    grp = df['items']
    print(1-df.groupby(grp)['mse'].sum()/df.groupby(grp)['ret_m'].sum())
    print(df.groupby(grp)['mse'].mean())


    df['acc']=np.sign(df['pred'])==np.sign(df['ret_m'])
    print(df.groupby(grp)['acc'].mean().sort_values())
    print(df['acc'].mean())



# 150k
# 30k