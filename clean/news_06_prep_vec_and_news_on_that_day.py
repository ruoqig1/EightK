import gc

import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from itertools import chain

def print_size(df,step):
    print(f'### Step {step}')
    print(df.groupby('alert')['body'].count()/1e6,flush=True)

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
    # for f in ['third']:
    for f in ['ref','third']:
        print('Start working on ',f)
        df = pd.read_pickle(start_dir + f+'.p')

        # format the date
        df['date'] = pd.to_datetime(df['timestamp'].apply(lambda x: x.split('T')[0]))
        # getting the tickers and market

        df['alert'] = df['body'].apply(len) == 0
        df =df.reset_index(drop=True)

        print_size(df,f'Full, {f}')


        #BUILD THE SOME NEWS DATAFRAME
        print('Start building some news')
        # some_news = df.apply(lambda row: {ticker:row['timestamp'] for ticker in row['ticker']},axis =1)
        some_news = df.apply(lambda row: [[ticker,row['timestamp'],row['id'],row['alert']] for ticker in row['ticker']],axis =1)
        print('Start merging the dict')
        # some_news = dict(chain.from_iterable(d.items() for d in some_news))
        some_news = list(chain.from_iterable(some_news))
        some_news = pd.DataFrame(some_news,columns=['ticker','timestamp','id','alert'])
        some_news.to_pickle(data.p_some_news_dir+f+'.p')
        # set it to none to save memory
        some_news = None
        gc.collect()

        # start building the vectorisation dataframe
        # keep only news with single stock
        ind = df['ticker'].apply(len)==1
        df =df.loc[ind,:]
        print_size(df,f'Keep only single stock {f}')
        print('Remaining articles')
        # keep only stock where we have a match for that day on crsp
        df['ticker'] =df['ticker'].apply(lambda x: x[0])
        df = df.merge(crsp[['date','ticker']].drop_duplicates(),how='inner')
        print_size(df,f'Merge with crsp {f}')

        # df['body'] = df['headline']+' \n '+df['body']
        df['year'] = df['date'].dt.year
        for y in tqdm.tqdm(np.unique(df['year'].values),'Save year per year'):
            df.loc[df['year']==y,['id','timestamp','date','headline','body','ticker','alert']].to_pickle(save_dir+f+f'{y}.p')
        print(f'Done for {f}',flush=True)
        print(df.head())

