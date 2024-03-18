import smtplib

import numpy as np
import pandas as pd
import os
from parameters import *
from matplotlib import pyplot as plt
from data import *
from utils_local.nlp_ticker import *
from statsmodels import api as sm
from joblib import Parallel, delayed
from tqdm import tqdm


def load_and_preprocess_v2(par):
    data = Data(par)
    df =data.load_crsp_all()
    df= df.sort_values(['permno', 'date']).reset_index(drop=True)
    # load the event to comput
    per = data.load_some_relevance_icf()
    per['date'] = pd.to_datetime(per['adate'])
    per = per.dropna(subset='date')
    per = per.loc[per['date'].dt.year>=2004,:]
    per['permno']=per['permno'].astype(int)
    ev = per[['date','permno','form_id']].dropna().drop_duplicates()
    ev['ev']= True


    # load ff and merge
    ff = data.load_ff5()
    df = df.merge(ff)
    df['ret']-=df['rf']
    df = df.merge(ev,how='left')
    # df['mktrf']=np.random.normal(size=df.shape[0])
    df['ev'] = df['ev'].fillna(False)
    # random event check to debug
    # df['ev'] = (np.random.uniform(size=df.shape[0])>0.99)

    print(df['ev'].sum(),ev.shape,df['ev'].mean(),flush=True)

    df = df.sort_values(['permno','date'])
    df = df.reset_index(drop=True)
    df['one'] = 1.
    return df



def process_group(gp,remove_alpha = False):
    res_list = []

    for i in gp[1].index[gp[1]['ev']]:
        train = gp[1].loc[(i - rolling_window - gap_window):(i - gap_window), ['ret'] + mkt_col]
        test = gp[1].loc[(i - ev_window):(i + ev_window), ['ret'] + mkt_col]

        if (test.shape[0] >= min_test) & (train.shape[0] >= min_rolling):
            m = sm.OLS(train[['ret']], train[mkt_col]).fit()
            if remove_alpha:
                abret = (m.params['one'] + test['ret'] - m.predict(test[mkt_col])).reset_index(drop=True)
            else:
                abret = (test['ret'] - m.predict(test[mkt_col])).reset_index(drop=True)
            abret.index = abret.index - ev_window
            abret = abret.reset_index().rename(columns={'index': 'evttime', 0: 'abret'})
            abret['ret'] = test['ret'].values
            abret['date'] = gp[1].loc[i]['date']
            abret['permno'] = gp[0]
            abret['sigma_ra'] = np.std(m.resid)
            abret['sigma_ret_train'] = np.std(train['ret'].values)
            abret['sigma_abs_ra'] = np.std(np.abs(m.resid))
            abret['beta'] = m.params['mktrf']
            res_list.append(abret)

    if res_list:
        return pd.concat(res_list, axis=0)
    return None


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    df = load_and_preprocess_v2(par)

    df.memory_usage(deep=True).sum() / (1024 ** 3)
    df.shape[0]/1e6

    ev_window = 20
    gap_window = 50
    rolling_window = 100
    min_rolling = 70
    min_test = 41

    # older versions that were runs
    # mkt_col = ['mktrf', 'one']
    # name = 'abn_ev_m.p'
    #
    # mkt_col = ['mktrf']
    # name = 'abn_ev_monly.p'
    #
    # mkt_col = ['mktrf','smb','hml','rmw','cma','umd']
    # name = 'abn_ev6_monly.p'
    #
    # mkt_col = ['mktrf','smb','hml']
    # name = 'abn_ev3_monly.p'

    # running current version
    mkt_col = ['mktrf','smb','hml','rmw','cma','umd','one']
    name = 'abn_ev6_long.p'

    # running current version check rf
    # mkt_col = ['mktrf','smb','hml','rmw','cma','umd','one']
    # name = 'abn_ev67_check_rf.p'



    n_jobs = -1  # Number of cores to use, -1 means all cores

    # Using tqdm to monitor the progress
    groups = list(df.groupby('permno'))

    results = Parallel(n_jobs=n_jobs)(delayed(process_group)(gp) for gp in tqdm(groups, desc="Processing groups"))
    # results = []
    # for gp in tqdm(groups, desc="Processing groups"):
    #     result = process_group(gp)
    #     results.append(result)

    res = pd.concat([r for r in results if r is not None], axis=0)

    res['abs_abret'] = res['abret'].abs()
    res.to_pickle(data.p_dir + name)

   # res = pd.read_pickle(data.p_dir + name)


    print(res)
    print('avg beta',res['beta'].mean())

    res.groupby('evttime')['abret'].mean().cumsum().plot()
    plt.show()
