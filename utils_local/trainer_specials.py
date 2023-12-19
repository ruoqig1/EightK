import sys
import pandas as pd
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
# from didipack.utils_didi.ridge import run_efficient_ridge
import numpy as np
from typing import Tuple
from parameters import *
from sklearn.linear_model import LogisticRegression



class TrainerLogisitcWithNewsInSample(TrainerLogisticElasticNet):
    def __init__(self, par: Params,para=None):
        super().__init__(par,para=para)
        self.m = None
        self.par = par

    def _split_and_index(self, x: pd.DataFrame, y: pd.DataFrame, ids: pd.Series, times: pd.Series, start, end,no_news = False) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        if no_news:
            indices = (times >= start) & (times < end) & (ids.apply(lambda x: float(x.split('-')[1]))==1.0)
        else:
            indices = (times >= start) & (times < end)
        x_sub = x.loc[indices, :]
        y_sub = y.loc[indices, :]
        ids_sub = ids.loc[indices]
        times_sub = times.loc[indices]
        y_sub = y_sub.set_index([times_sub, ids_sub])
        print('$$$$$$$$$$$$ HERE',flush=True)
        return x_sub, y_sub

    def _split_data(self, x: pd.DataFrame, y: pd.DataFrame, ids: pd.Series, times: pd.Series, t_index) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.par.train.T_train < 0:
            start_train = times.min()
        else:
            start_train = t_index - self.par.train.T_train


        end_train = t_index - self.par.train.T_val
        end_test = t_index + self.par.train.testing_window
        x_train, y_train = self._split_and_index(x, y, ids, times, start_train, end_train,no_news=True)
        x_val, y_val = self._split_and_index(x, y, ids, times, end_train, t_index,no_news=True)
        x_test, y_test = self._split_and_index(x, y, ids, times, t_index, end_test, no_news=False)
        return x_train, y_train, x_val, y_val, x_test, y_test


    def _validation_procedure(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame,
                              y_val: pd.DataFrame, hyper_parameters_set):

        # if no grid is defiend, default is 0.5
        try:
            list_of_l1_ratio= self.par.train.l1_ratio
        except:
            list_of_l1_ratio = [0.5]

        perf = {}
        comb = {}

        k=0
        for l1_ratio in list_of_l1_ratio:
            for shrinkage in self.par.train.shrinkage_list:
                hyp = {'shrinkage':[shrinkage],'l1_ratio':[l1_ratio]}
                print('Just before the analysis',flush=True)
                self._train_model(x=x_train,y=y_train,hyper_params=hyp)
                perf[k] = self.m.score(X=x_val,y=y_val)
                comb[k]=hyp
                k+=1
                print('Ran One analysis',k,flush=True)

        best_hype = comb[pd.Series(perf).idxmax()]
        print('Select best hype', best_hype,flush=True)
        return best_hype

if __name__ == "__main__":
    try:
        grid_id = int(sys.argv[1])
        model_id = int(sys.argv[2])
        print('Running with args', grid_id, flush=True)
    except:
        print('Debug mode on local machine', flush=True)
        grid_id = 0
        model_id = 5

    par = Params()
    par.train.testing_window = 5
    self = TrainerLogisticElasticNet(par)

    N = 1000
    P = 100
    np.random.seed(1)
    fake_data = pd.DataFrame(np.random.normal(size=(N,P)))
    y = pd.DataFrame(np.sign(fake_data.mean(1)))
    dates = pd.Series(np.arange(N))
    ids = dates.copy()

    y_test, _ = self.train_at_time_t(x=fake_data,y=y,ids=ids,times=dates,t_index_for_split=100)

