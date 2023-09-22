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

    f = 'sample.p'
    for f in ['ref','third']:
        print('Start working on ',f)
        df = pd.read_pickle(start_dir + f+'.p')

        # format the date
        df['date'] = pd.to_datetime(df['timestamp'].apply(lambda x: x.split('T')[0]))

        # keep only single tickers
        ind = df['ticker'].apply(len)==1
        df = df.loc[ind,:]
        df['ticker'] = df['ticker'].apply(lambda x: str(x[0]).upper())

        # merge with crsp to keep only text with some degree of match in real data.
        df= df.merge(crsp[['date','ticker']])

        # store only appropriate data of appropriate sizes
        df['txt'] = df['headline']+' \n \n ' +df['body']
        df=df[['id','date','ticker','txt']].drop_duplicates()
        years = df['date'].dt.year
        for y in np.sort(np.unique(years.values)):
            save_path = save_dir+f'{f}_{y}.p'
            df.loc[years==y,:].to_pickle(save_path)
            print('saved in',save_path, flush=True)
