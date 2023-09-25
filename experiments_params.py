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
from utils_local.trainer_specials import *


def get_main_experiments(id_comb:int,train=True) -> Params:
    print('START WORKING ON ',id_comb,flush=True)
    par=Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL

    par.train.testing_window = 12
    par.train.T_val = 6

    par.train.min_nb_chunks_in_cluster=1

    par.train.nb_chunks=14
    # par.train.pred_model = PredModel.LOGIT_EN
    # par.train.norm = Normalisation.ZSCORE

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.norm = Normalisation.ZSCORE
    par.train.tnews_only = True

    par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]

    year_id_list = np.arange(0,15,1) if train else [0]
    grid = [
        ['train', 'T_train',[-60,60]],
        ['grid', 'year_id',year_id_list],
        ['train', 'norm', [Normalisation.ZSCORE]], # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.0], [0.5],[1.0]]]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par



# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/'


