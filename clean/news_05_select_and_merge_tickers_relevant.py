import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *

'''
Select all the news with at least one tickers in INVALID MARKET, New york stock exchange (N) or Nasdaq (O)
'''


if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    final = pd.DataFrame()

    start_dir = data.p_news_year if args.a==1 else data.p_news_third_party_year
    dest_dir =  data.p_news_tickers_related+'ref.p' if args.a==1 else data.p_news_tickers_related+'third.p'
    print('Start working on ',start_dir)
    print('To save in',dest_dir)


    for f in tqdm.tqdm(os.listdir(start_dir),'Dropping and merging'):
        df = pd.read_pickle(start_dir+f).reset_index(drop=True)
        list_valid = data.load_list_of_tickers_in_news_and_crsp()
        list_valid = list(list_valid.drop_duplicates().values)

        # getting the tickers and market
        tickers_and_market = df['subjects'].apply(lambda x: [xx.replace('R:', '') for xx in x if 'R:' in xx])
        # splitting ticker and market
        tickers = tickers_and_market.apply(lambda x: [clean_ticker(xx, list_valid=list_valid) for xx in x])
        # merging into a df to use apply
        markets = tickers_and_market.apply(lambda x: [get_market(xx) for xx in x])
        tickers = tickers.reset_index().rename(columns={'subjects': 'ticker'})
        tickers['market'] = markets.values
        # building the lambda function
        def func(x):
            list_to_keep = []
            for i in range(len(x['ticker'])):
                if x['market'][i] in ['N', 'O', 'INVALID_MARKET']:
                    list_to_keep.append(x['ticker'][i])
            return list_to_keep
        # getting the ticker that are only
        tickers['res'] = tickers.apply(func, axis=1)
        tickers = tickers['res'].apply(lambda x: [xx for xx in x if xx != ''])
        # keep only when we have at least one ticker.
        ind=tickers.apply(len)>0
        df=df.loc[ind,:]
        tickers=tickers.loc[ind]

        df['ticker']=tickers
        drop_cols = ['pubStatus','mimeType','altId','source','name']
        df=df.drop(columns=drop_cols)
        final = pd.concat([final,df],axis=0)
    final.to_pickle(dest_dir)
    print('Saved',final)
    print(final.shape,flush=True)


