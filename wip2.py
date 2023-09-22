import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *

def df_size(df):
    memory_in_gb = df.memory_usage(deep=True).sum() / 1e9
    return memory_in_gb

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    final = pd.DataFrame()

    start_dir = data.p_news_year if args.a==1 else data.p_news_third_party_year
    dest_dir =  data.p_news_tickers_related+'ref.p' if args.a==1 else data.p_news_tickers_related+'third.p'

    df = pd.read_pickle(dest_dir)

    ind=df['ticker'].apply(len)
    (ind==1).sum()

    ind = df['subjects'].apply(lambda x: 'N2:US' in x)
    df['date']=pd.to_datetime(df['timestamp'].apply(lambda x: x.split('T')[0]))
    year = df['date'].dt.year

    ind.groupby(year).sum()

