import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates,get_chunks,get_tasks_for_current_node
import psutil
from utils_local.trainer_specials import *
from experiments_params import get_main_experiments
def set_ids_to_eight_k_df(df:pd.DataFrame,par:Params):
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
    if par.train.tnews_only is not None:
        df['id']=df['id'].astype(str)+'-'+df['news0'].astype(str)
    return df


def generate_fake_data(N: int = 1000, P: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generates a dataset of fake data for testing purposes.

    Args:
        N (int): The number of rows in the generated data.
        P (int): The number of columns in the generated data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the fake data, its mean along the columns, a series representing dates, and a series representing ids.
    """
    fake_data = pd.DataFrame(np.random.normal(size=(N, P)))
    y = pd.DataFrame(np.sign(fake_data.mean(1)))
    dates = pd.Series(np.arange(N))
    ids = dates.copy()

    return fake_data, y, dates, ids

def chose_trainer(par:Params):
    m = None
    if par.train.pred_model == PredModel.RIDGE:
        m = TrainerRidge(par)
    if par.train.pred_model == PredModel.LOGIT_EN:
        if par.train.tnews_only is None:
            m = TrainerLogisticElasticNet(par)
        else:
            m = TrainerLogisitcWithNewsInSample(par,para=-1)
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
    print('START WORKING ON ',args.a,flush=True)
    args.a = 1
    par = get_main_experiments(args.a,train=True)
    nb_nodes_total = 14
    print('Parameter defined',flush=True)
    par.print_values()

    # for i in range(90):
    for i in [0]:
        par = get_main_experiments(i, train=True)
        temp_save_dir = par.get_res_dir()
        print(temp_save_dir)
    # 0 NOW
    # save /data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio0.5nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p

    if socket.gethostname()=='3330L-214940-M':
        temp_dir = 'res/temp_data/'
        os.makedirs(temp_dir,exist_ok=True)
        x, y, dates, ids = generate_fake_data()
        start_id = 100
        par.train.tnews_only = None

    else:
        load_dir = par.get_training_dir()
        print('Start loading Df',flush=True)
        df= pd.read_pickle(load_dir+'main_df.p')
        print('Loaded Df',flush=True)
        df = set_ids_to_eight_k_df(df,par)
        print('set ids',flush=True)
        x = np.load(load_dir+'x.npy')
        x = pd.DataFrame(x)
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



    trainer = chose_trainer(par)
    verbose = True
    temp_save_dir = par.get_res_dir()
    print(temp_save_dir)

    start_dates = get_start_dates(dates, par.train.T_train, par.train.testing_window)
    chunks_still_to_process, chunks_already_processed = get_chunks(start_dates,  par.train.nb_chunks, temp_save_dir)



    if verbose:
        print(f'Already processed {len(chunks_already_processed)}, Still to processed {len(chunks_still_to_process)}',flush=True)

    to_run_now = get_tasks_for_current_node(chunks_still_to_process, nb_nodes_total, par.train.min_nb_chunks_in_cluster, par.grid.year_id)
    k = 0

    for chunk in to_run_now:
        k+=1
        df_oos_pred = pd.DataFrame()
        for start_id in tqdm.tqdm(chunk[1],f'Chunks {k} ({chunk[0]}) out of {len(to_run_now)}'):
            y_test, _ = trainer.train_at_time_t(x=x, y=y, ids=ids, times=dates, t_index_for_split=start_id)
            y_test, _ = trainer.train_at_time_t(x=x, y=y, ids=ids, times=dates, t_index_for_split=start_id)
            df_oos_pred = pd.concat([df_oos_pred, y_test], axis=0)
        df_oos_pred.to_pickle(temp_save_dir + chunk[0])
        print(df_oos_pred.head(),0,flush=True)
        print('save', temp_save_dir + chunk[0])

    if par.grid.year_id == 0:
        par.save_model_params_in_main_file()


