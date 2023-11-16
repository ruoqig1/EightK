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


def load_and_preprocess_volume_data(par):
    data = Data(par)
    df =data.load_turnover()
    df= df.sort_values(['permno', 'date']).reset_index(drop=True)
    # load the event to comput
    per = data.load_some_relevance_icf()
    per['date'] = pd.to_datetime(per['adate'])
    per = per.dropna(subset='date')
    per = per.loc[per['date'].dt.year>=2004,:]
    per['permno']=per['permno'].astype(int)
    ev = per[['date','permno']].dropna().drop_duplicates()
    ev['ev']= True


    # load ff and merge
    df = df.merge(ev,how='left')
    df['ev'] = df['ev'].fillna(False)
    print(df['ev'].sum(),ev.shape,df['ev'].mean(),flush=True)
    df = df.sort_values(['permno','date'])
    df = df.reset_index(drop=True)
    df['one'] = 1.0
    return df


def process_group(gp,remove_alpha = False):
    res_list = []


    for i in gp[1].index[gp[1]['ev']]:
        train = gp[1].loc[(i - rolling_window - gap_window):(i - gap_window), ['turnover'] + mkt_col]
        test = gp[1].loc[(i - ev_window):(i + ev_window), ['turnover'] + mkt_col]

        if (test.shape[0] >= min_test) & (train.shape[0] >= min_rolling):
            m = sm.OLS(train[['turnover']], train[mkt_col]).fit()
            if remove_alpha:
                abret = (m.params['one'] + test['turnover'] - m.predict(test[mkt_col])).reset_index(drop=True)
            else:
                abret = (test['turnover'] - m.predict(test[mkt_col])).reset_index(drop=True)
            abret.index = abret.index - ev_window
            abret = abret.reset_index().rename(columns={'index': 'evttime', 0: 'abret'})
            abret['ret'] = test['turnover'].values
            abret['date'] = gp[1].loc[i]['date']
            abret['permno'] = gp[0]
            abret['sigma_ra'] = np.std(m.resid)
            abret['sigma_ret_train'] = np.std(train['turnover'].values)
            abret['sigma_abs_ra'] = np.std(m.resid)
            # abret['sigma_abs_ra'] = np.std(np.abs(m.resid))
            res_list.append(abret)

    if res_list:
        return pd.concat(res_list, axis=0)
    return None


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    df = load_and_preprocess_volume_data(par)

    ev_window = 20
    gap_window = 50
    rolling_window = 100
    min_rolling = 70
    min_test = 41

    mkt_col = ['turnover_m', 'one']
    name = 'turn_ev_m.p'
    mkt_col = ['turnover_m']
    name = 'trun_ev_monly.p'

    n_jobs = -1  # Number of cores to use, -1 means all cores
    # Using tqdm to monitor the progress
    groups = list(df.groupby('permno'))
    print('new again')

    results = Parallel(n_jobs=n_jobs)(delayed(process_group)(gp) for gp in tqdm(groups, desc="Processing groups"))

    res = pd.concat([r for r in results if r is not None], axis=0)


    res.to_pickle(data.p_dir + name)
    print('saved to', data.p_dir + name)

