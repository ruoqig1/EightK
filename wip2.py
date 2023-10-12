import smtplib

import numpy as np
import pandas as pd
import os
from parameters import *
from matplotlib import pyplot as plt
from data import *
from utils_local.nlp_ticker import *
from statsmodels import api as sm

def load_and_preprocess(par):
    data = Data(par)
    df =data.load_crsp_all()
    # load the event to comput
    per = data.load_some_relevance_icf()
    per['date'] = pd.to_datetime(per['adate'])
    per = per.dropna(subset='date')
    per = per.loc[per['date'].dt.year>=2004,:]
    per['permno']=per['permno'].astype(int)
    ev = per[['date','permno']].dropna().drop_duplicates()
    ev['ev']= True

    # load ff and merge
    ff = data.load_ff5()
    df = df.merge(ff)
    df['ret']-=df['rf']
    df = df.merge(ev,how='left')
    df['ev'] = df['ev'].fillna(False)
    print(df['ev'].sum(),ev.shape,flush=True)

    df = df.sort_values(['permno','date'])
    df = df.reset_index(drop=True)
    df['one'] = 1.0
    return df



if __name__ == "__main__":
    par = Params()
    data = Data(par)

    df = data.load_crsp_daily()
    df['ret'] = pd.to_numeric(df['ret'],errors='coerce')
    df['year'] = df['date'].dt.year
    df=df.merge(data.load_mkt_cap_yearly())
    df = df.loc[df['mcap_d']>=8,:]

    t=df.groupby('permno')['ret'].aggregate(['mean','std'])
    t['mean']*=20
    t['std']*=np.sqrt(20)
    t.median()




    # os.listdir(data.p_eight_k_clean)
    # df = pd.read_pickle(data.p_eight_k_clean+'legal_2006.p')
    # df = df.reset_index(drop=True)
    #
    # pyperclip.copy(df.loc[10000,'txt'])