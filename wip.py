import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
# from utils_local.nlp_ticker import *
# from didipack.utils_didi.ridge import run_efficient_ridge
# from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
# from utils_local.trainer_logistic_elastic_net import TrainerLogisticElasticNet
# from didipack.trainer.train_splitter import get_start_dates,get_chunks,get_tasks_for_current_node
import psutil
from utils_local.trainer_specials import *

if __name__ == "__main__":
    start_dir ='res/list_usage2/'

    df = pd.read_pickle(start_dir+'df_0.p')


    df.dropna()