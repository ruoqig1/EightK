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


    for f in tqdm.tqdm(os.listdir(data.p_news_year),'Dropping and merging'):
        df = pd.read_pickle(data.p_news_year+f)
        list_valid = data.load_list_of_tickers_in_news_and_crsp()
        list_valid = list(list_valid.drop_duplicates().values)
        tickers = df['subjects'].apply(lambda x: [xx.replace('R:', '') for xx in x if 'R:' in xx])
        # reducing the sample on raw tickers to save computing time in the big apply
        ind=tickers.apply(len)
        df=df.loc[ind>0,:]
        tickers=tickers.loc[ind>0]
        tickers=tickers.apply(lambda x: [clean_ticker(xx,list_valid=list_valid) for xx in x])
        tickers=tickers.apply(lambda x: [xx for xx in x if xx!=''])
        # reducing the tickers to final size now
        ind = tickers.apply(len)
        df = df.loc[ind > 0, :]
        tickers = tickers.loc[ind > 0]
        df['ticker']=tickers
        drop_cols = ['pubStatus','mimeType','altId','provider','source','name']
        df=df.drop(columns=drop_cols)
        final = pd.concat([final,df],axis=0)
    final.to_pickle(dest_dir)
    print('Saved',final)
    print(final.shape,flush=True)
    print(df_size(final),flush=True)

