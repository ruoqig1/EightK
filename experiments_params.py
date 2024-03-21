from parameters import *
import numpy as np
import pandas as pd


def get_main_experiments(id_comb: int, train=True, train_gpu=False, tensorboard=False) -> Params:
    print('START WORKING ON ', id_comb, flush=True)
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL_ATI_TRAIN

    par.train.testing_window = 1
    par.train.T_val = 2
    par.train.tensorboard = tensorboard

    par.train.min_nb_chunks_in_cluster = 1

    par.train.pred_model = PredModel.LOGIT_EN

    par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2012, 2023, 1) if train else [0]

    par.train.tnews_only = True
    grid = [
        ['train', 'T_train', [6]],
        ['train', 'norm', [Normalisation.ZSCORE, Normalisation.MINMAX]],  # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.0], [0.5], [1.0]]],
        ['train', 'abny', [True, None]],
        ['train', 'news_filter_training', ['news0', 'news_with_time']],
        ['grid', 'year_id', year_id_list]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


def get_main_experiments_train_all(id_comb:int,train=True,train_gpu = False) -> Params:
    print('START WORKING ON ', id_comb, flush=True)
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL_ATI_TRAIN

    par.train.testing_window = 1
    par.train.T_val = 2

    par.train.min_nb_chunks_in_cluster = 1

    par.train.pred_model = PredModel.LOGIT_EN

    par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2012, 2023, 1) if train else [0]


    par.train.tnews_only = False
    par.train.news_filter_training = None
    grid = [
        ['train', 'T_train', [8]],
        ['train', 'norm', [Normalisation.ZSCORE,Normalisation.MINMAX]],  # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.0], [0.5], [1.0]]],
        ['train', 'abny', [True, None]],
        ['grid', 'year_id', year_id_list]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


def get_first_tf_training(id_comb:int,train=True,train_gpu = False) -> Params:
    print('START WORKING ON ',id_comb,flush=True)
    par=Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.NEWS_SINGLE

    par.train.testing_window = 1
    par.train.T_val = 2

    par.train.min_nb_chunks_in_cluster = 1

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.tnews_only = False
    par.train.batch_size = 512
    par.train.use_tf_models = True
    par.train.adam_rate = 0.001
    par.train.patience = 3  # 5
    par.train.monitor_metric = 'loss'  # ='loss'
    par.train.max_epoch = 100  # ='loss'
    par.train.train_on_gpu = train_gpu
    # par.train.l1_ratio = [0.0] # 0.0 for ridge (which is what these assholes do....)
    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    par.train.shrinkage_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2004, 2023, 1) if train else [0]
    par.train.apply_filter = None
    par.train.filter_on_cosine = None
    par.train.filter_on_alert = None
    par.train.filter_on_prn = None
    par.train.filter_on_reuters = None
    par.train.abny = 'ret_m'
    grid = [
        ['train', 'T_train', [8]],
        ['train', 'norm', [Normalisation.NO, Normalisation.ZSCORE]],  # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.5], [0.0]]],
        ['grid', 'year_id', year_id_list],
        ['train', 'adam_rate', [0.001, 0.01]]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


def get_experiments_coverage_pred(id_comb: int, train=True) -> Params:
    print('START WORKING ON ', id_comb, flush=True)
    par = Params()
    grid = [
        ['covpred', 'predictors',
         [PredictorsCoverage.ITEMS_NAMES_AND_SIZE, PredictorsCoverage.ALL, PredictorsCoverage.COVE_ONLY,
          PredictorsCoverage.ALL_BUT_COV]],
        # ['covpred', 'small_sample',[False,True]],
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


def get_experiment_for_paper_draft_in_oct_2023(id_comb: int, train=True) -> Params:
    print('START WORKING ON ', id_comb, flush=True)
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL

    par.train.testing_window = 1
    par.train.T_val = 2

    par.train.min_nb_chunks_in_cluster = 1

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.tnews_only = False

    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    par.train.shrinkage_list = [0.001, 0.01, 0.1, 1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2012, 2023, 1) if train else [0]

    grid = [
        ['train', 'T_train', [8]],
        ['train', 'norm', [Normalisation.ZSCORE]],  # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.0], [0.5], [1.0]]],
        ['train', 'abny', [True, False]],
        ['grid', 'year_id', year_id_list]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


def get_params_for_tfidf():
    par = Params()
    par.enc.opt_model_type = OptModelType.BOW1
    par.tfidf.no_above = 1
    par.tfidf.no_below = .99
    par.tfidf.do_some_filtering = False
    par.tfidf.vocabulary_list = VocabularySetTfIdf.REUTERS_ONLY
    return par


def predict_with_news_based_on_some_filters(id_comb: int, train=True, train_gpu=False) -> Params:
    print('START WORKING ON ', id_comb, flush=True)
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.NEWS_REF_ON_EIGHT_K

    par.train.testing_window = 1
    par.train.T_val = 2

    par.train.min_nb_chunks_in_cluster = 1

    par.train.pred_model = PredModel.LOGIT_EN
    par.train.tnews_only = False
    par.train.batch_size = 512
    par.train.use_tf_models = True
    par.train.adam_rate = 0.001
    par.train.patience = 3  # 5
    par.train.monitor_metric = 'loss'  # ='loss'
    par.train.max_epoch = 100  # ='loss'
    par.train.train_on_gpu = train_gpu
    par.train.l1_ratio = [0.5]
    # par.train.shrinkage_list = [0.001, 0.01, 0.1, 1, 10]
    par.train.shrinkage_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    # par.train.shrinkage_list = [1.0]

    # year_id_list = np.arange(0,15,1) if train else [0]
    year_id_list = np.arange(2012, 2023, 1) if train else [0]
    par.train.apply_filter = ['val', 'train']
    par.train.filter_on_cosine = None
    par.train.filter_on_alert = None
    par.train.filter_on_prn = None
    par.train.filter_on_reuters = None
    grid = [
        ['train', 'T_train', [8]],
        ['train', 'norm', [Normalisation.ZSCORE]],  # ,Normalisation.MINMAX
        ['train', 'l1_ratio', [[0.5]]],
        ['train', 'abny', [True, False]],
        ['grid', 'year_id', year_id_list]
    ]
    # par.train.T_train = -60
    par.update_param_grid(grid, id_comb=id_comb)
    return par


if __name__ == '__main__':

    get_main_experiments(0,True)
    get_main_experiments(0,False)
    get_main_experiments_train_all(0,True)
    get_main_experiments_train_all(0,False)
    par = get_main_experiments_train_all(1,False)

    par.print_values()

# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/chunk_0.p
# save ./res/temp/vec_pred/T_train-60T_val6testing_window12shrinkage_list5pred_modelPredModel.LOGIT_ENnormNormalisation.ZSCOREsave_insFalsetnews_onlyTruel1_ratio1nb_chunks14min_nb_chunks_in_cluster1/OPT_13b/EIGHT_LEGAL/'
