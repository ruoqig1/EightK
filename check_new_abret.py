import smtplib

import numpy as np
import pandas as pd
import os
from parameters import *
from matplotlib import pyplot as plt
from data import *
from utils_local.nlp_ticker import *
from statsmodels import api as sm

def df_size(df):
    memory_in_gb = df.memory_usage(deep=True).sum() / 1e9
    return memory_in_gb

if __name__ == "__main__":
    par = Params()
    data = Data(par)

    per = data.load_list_by_date_time_permno_type()
    per['date'] = pd.to_datetime(per['fdate'])
    per = per.dropna(subset='date')
    per = per.loc[per['date'].dt.year>=2004,:]
    per['permno']=per['permno'].astype(int)

    res = pd.read_pickle(data.p_dir+'ev.p')
    # res = data.load_e_ff_long(window=40, tp='m', reload=False).drop(columns='date').rename(columns={'evtdate': 'date'})



    rav = data.load_ravenpack_all()

    ind = rav['rtime'].apply(lambda x: int(x[:2]) <= 16)
    rav = rav.loc[ind, :]
    rav['news0'] = (rav['relevance'] >= 1) * 1
    rav = rav.groupby(['rdate', 'permno'])[['news0']].max().reset_index()
    rav = rav.rename(columns={'rdate': 'date'})
    rav['permno'] = rav['permno'].astype(int)

    # remove the event before the tiem
    # per['hours'] = per['ftime'].apply(lambda x: int(str(x)[:2]))

    res
    rav
    df = per[['permno','items','date']].merge(rav,how='left')
    df['news0'] = df['news0'].fillna(0.0)
    df = df.merge(res)

    df['abs_trim'] = PandasPlus.winzorize_series(df['abret'],1).abs()
    df.groupby('evttime')['abs_trim'].mean().plot()
    plt.show()

    df['year'] = df['date'].dt.year

    df=df.merge(data.load_mkt_cap_yearly())
    df['large_firm'] = df['mcap_d']==10

    grp_col = 'news0'

    for large_firm in [True,False]:
        ind = df['large_firm']==large_firm
        df.loc[ind,:].groupby(['evttime',grp_col])['abs_trim'].mean().reset_index().pivot(columns=grp_col,index='evttime',values='abs_trim').plot()
        plt.title(f'Large firm {large_firm}')
        plt.show()

    # per
    # Index(['URL', 'acceptanceDatetime', 'accessionNumber', 'type', 'publicDocumentCount', 'period', 'items', 'filingDate',
    # 'dateOfFilingDateChange', 'sros', 'name', 'cik', 'sic', 'IRSNumber', 'stateOfIncorporation', 'fiscalYearend', 'formType',
    # 'act', 'fileNumber', 'filmNumber', 'businessStreet1', 'businessStreet2', 'businessCity', 'businessState', 'businessZip',
    # 'businessPhone', 'mailingStreet1', 'mailingStreet2', 'mailingCity', 'mailingState', 'mailingZip', 'formerName', 'dateChanged',
    # 'atime', 'adate', 'fdate', 'first_date', 'permno', 'gvkey', 'date'], dtype='object')
    # per['atime']
