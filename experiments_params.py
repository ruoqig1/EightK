from parameters import *
import numpy as np
import pandas as pd


def get_main_experiments(id_comb:int,train=True) -> Params:
    print('START WORKING ON ',id_comb,flush=True)
    par=Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL

    par.train.testing_window = 1
    par.train.T_val = 2

    par.train.min_nb_chunks_in_cluster=1

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.tnews_only = True

    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    par.train.shrinkage_list = [0.001, 0.01, 0.1, 1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2012,2023,1) if train else [0]

    grid = [
        ['train', 'T_train',[8]],
        ['train', 'norm', [Normalisation.ZSCORE]], # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.0], [0.5],[1.0]]],
        ['train', 'abny', [True,False]],
        ['grid', 'year_id', year_id_list]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


if __name__ == '__main__':
    get_main_experiments(0)

# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/'

