import smtplib

import numpy as np
import pandas as pd
import os
from parameters import *
from matplotlib import pyplot as plt
from data import *
from utils_local.nlp_ticker import *
from statsmodels import api as sm

def load_res_with_crsp():
    crsp = data.load_crsp_all().merge(rel[['permno']].drop_duplicates())
    crsp = crsp.sort_values(['permno', 'date']).reset_index(drop=True)
    col_lag = []
    for i in tqdm.tqdm(np.arange(-20, 21, 1)):
        n = i
        crsp[n] = crsp.groupby('permno')['ret'].transform('shift', -i)
        col_lag.append(n)

    res = rel.merge(crsp[['permno', 'date'] + col_lag])

    res = res.melt(id_vars=['permno', 'date'], value_vars=col_lag).reset_index()
    res = res.rename(columns={'variable': 'evttime', 'value': 'ret'})
    return res

if __name__ == "__main__":
    par = Params()
    data = Data(par)

    rel = data.load_some_relevance_icf().rename(columns={'adate':'date'})

    # res = data.load_e_ff_long(window=40, tp='m', reload=False).drop(columns='date').rename(columns={'evtdate': 'date'})
    # res = load_res_with_crsp()
    # res = pd.read_pickle(data.p_dir+'ev_test.p')
    res = pd.read_pickle(data.p_dir+'abn_ev_monly.p')
    # res['abret_abs'] = PandasPlus.winzorize_series(res['abret'],1).abs()
    res['abret_abs'] = res['abret'].abs()
    res['ret_abs'] = PandasPlus.winzorize_series(res['ret'],1).abs()
    # res['ret_abs'] = PandasPlus.winzorize_series(res['ret'],1).abs()

    res =res.merge(rel)

    res['year'] = res['date'].dt.year
    res=res.merge(data.load_mkt_cap_yearly())
    res['large_firm'] = res['mcap_d'] == 10


    ret_col ='abret_abs'
    res.groupby('evttime')[ret_col].mean().plot()
    plt.show()


    grp_col = 'no_rel'
    std = 'sigma_ra'
    ind = ~res[['date','evttime','permno']].duplicated()
    res = res.loc[ind,:]

    for large_firm in [True,False]:
        ind = res['large_firm'] == large_firm
        m = res.loc[ind, :].groupby(['evttime', grp_col])[ret_col].mean().reset_index().pivot(columns=grp_col, index='evttime', values=ret_col)
        s = res.loc[ind, :].groupby(['evttime', grp_col])[std].mean().reset_index().pivot(columns=grp_col, index='evttime', values=std)
        s = np.sqrt(s)
        m.plot()
        plt.title(f'Large firm {large_firm}')
        plt.show()
