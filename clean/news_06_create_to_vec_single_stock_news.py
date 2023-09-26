import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    final = pd.DataFrame()

    start_dir = data.p_news_tickers_related
    save_dir = data.p_to_vec_main_dir+'/single_stock_news_to_vec/'
    os.makedirs(save_dir,exist_ok=True)
    crsp = data.load_crsp_daily()
    print('loaded CRSP')
    list_valid = data.load_list_of_tickers_in_news_and_crsp()
    list_valid = list(list_valid.drop_duplicates().values)
    f = 'sample.p'
    # for f in ['ref','third']:
    for f in ['sample']:
        print('Start working on ',f)
        df = pd.read_pickle(start_dir + f+'.p')

        # format the date
        df['date'] = pd.to_datetime(df['timestamp'].apply(lambda x: x.split('T')[0]))
        # getting the tickers and market
        tickers_and_market = df['subjects'].apply(lambda x: [xx.replace('R:', '') for xx in x if 'R:' in xx])
        # splitting ticker and market
        ticker = tickers_and_market.apply(lambda x: [clean_ticker(xx, list_valid=list_valid) for xx in x])
        # merging into a df to use apply
        market = tickers_and_market.apply(lambda x: [get_market(xx) for xx in x])
        ticker=ticker.reset_index().rename(columns={'subjects':'ticker'})
        ticker['market'] = market.values
        # building the lambda function
        def func(x):
            list_to_keep = []
            for i in range(len(x['ticker'])):
                if x['market'][i] in ['N','O','INVALID_MARKET']:
                    list_to_keep.append(x['ticker'][i])
            return list_to_keep
        # getting the ticker that are only
        ticker = ticker.apply(func,axis=1)
        ticker = ticker.apply(lambda x: [xx for xx in x if xx != ''])
        ind = ticker['res'].apply(len)==1

        # df['ticker']
        # keep only single tickers
        # ind = df['ticker'].apply(len)==1
        # df = df.loc[ind,:]
        # df['ticker'] = df['ticker'].apply(lambda x: str(x[0]).upper())

        # # merge with crsp to keep only text with some degree of match in real data.
        # df= df.merge(crsp[['date','ticker']])
        #
        # # store only appropriate data of appropriate sizes
        # df['txt'] = df['headline']+' \n \n ' +df['body']
        # df=df[['id','date','ticker','txt']].drop_duplicates()
        # years = df['date'].dt.year
        # for y in np.sort(np.unique(years.values)):
        #     save_path = save_dir+f'{f}_{y}.p'
        #     df.loc[years==y,:].to_pickle(save_path)
        #     print('saved in',save_path, flush=True)
