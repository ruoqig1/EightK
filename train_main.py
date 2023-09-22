import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from utils_local.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates,get_chunks,get_tasks_for_current_node
import psutil

def chose_trainer(par:Params):
    m = None
    if par.train.pred_model == PredModel.RIDGE:
        m = TrainerRidge(par)
    if par.train.pred_model == PredModel.LOGIT_EN:
        m = TrainerLogisticElasticNet(par)
    return m

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # memory in GB

def normalize(x, par:Params):
    if par.train.norm == Normalisation.ZSCORE:
        x = (x - x.mean()) / x.std()
    if par.train.norm == Normalisation.RANK:
        x = x.rank(pct=True,axis=1)-0.5
    if par.train.norm == Normalisation.MINMAX:
        x = 2 * ((x - x.min()) / (x.max() - x.min())) - 1
    return x

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL

    par.train.testing_window = 12
    par.train.T_val = 6
    par.train.T_train =-60
    par.train.min_nb_chunks_in_cluster=1
    nb_nodes_total = 14
    par.train.nb_chunks=14
    # par.train.pred_model = PredModel.LOGIT_EN
    # par.train.norm = Normalisation.ZSCORE

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.norm = Normalisation.ZSCORE

    if par.train.pred_model == PredModel.LOGIT_EN:
        par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    print('Parameter defined')

    if socket.gethostname()=='3330L-214940-M':
        temp_dir = 'res/temp_data/'
        os.makedirs(temp_dir,exist_ok=True)
        y = pd.read_pickle(temp_dir+'y.p')
        x = pd.read_pickle(temp_dir+'x.p')
        dates = pd.read_pickle(temp_dir+'dates.p')
        ids = pd.read_pickle(temp_dir+'ids.p')

    else:
        load_dir = par.get_training_dir()
        df= pd.read_pickle(load_dir+'main_df.p')
        df =df.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
        df=df.sort_values('date')
        x = pd.DataFrame(np.load(load_dir+'x.npy'))
        print('Start normalizing')
        x = normalize(x,par)
        print('Normalized',flush=True)

        dates = PandasPlus.get_ym(df['date'])
        dates_tr= dates.drop_duplicates().sort_values().reset_index(drop=True).reset_index().rename(columns={'index':'t_ind'})
        dates=dates.reset_index(drop=False).merge(dates_tr,how='left')['t_ind']

        y =df[['ret_m']]
        if par.train.pred_model == PredModel.LOGIT_EN:
            y = np.sign(y)
            y = y.replace({0:1})
        ids = df['id']
        print('Data loaded',flush=True)
        # temp_dir = 'res/temp_data/'
        # os.makedirs(temp_dir,exist_ok=True)
        # y.to_pickle(temp_dir+'y.p')
        # x.iloc[:,:10].to_pickle(temp_dir+'x.p')
        # dates.to_pickle(temp_dir+'dates.p')
        # ids.to_pickle(temp_dir+'ids.p')



    trainer = chose_trainer(par)
    verbose = True
    temp_save_dir = par.get_res_dir()

    start_dates = get_start_dates(dates, par.train.T_train, par.train.testing_window)
    chunks_still_to_process, chunks_already_processed = get_chunks(start_dates,  par.train.nb_chunks, temp_save_dir)



    if verbose:
        print(f'Already processed {len(chunks_already_processed)}, Still to processed {len(chunks_still_to_process)}',flush=True)

    to_run_now = get_tasks_for_current_node(chunks_still_to_process, nb_nodes_total, par.train.min_nb_chunks_in_cluster, args.a)
    k = 0

    for chunk in to_run_now:
        k+=1
        df_oos_pred = pd.DataFrame()
        for start_id in tqdm.tqdm(chunk[1],f'Chunks {k} ({chunk[0]}) out of {len(to_run_now)}'):
            y_test, _ = trainer.train_at_time_t(x=x, y=y, ids=ids, times=dates, t_index_for_split=start_id)
            df_oos_pred = pd.concat([df_oos_pred, y_test], axis=0)
        df_oos_pred.to_pickle(temp_save_dir + chunk[0])
        print('save', temp_save_dir + chunk[0])

    if args.a == 0:
        par.save_model_params_in_main_file()


