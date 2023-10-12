import gc

import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from itertools import chain
def load_all_index(par:Params, reload = True):
    sources = []
    for news_source in [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD]:
        par.enc.news_source = news_source
        if reload:
            print(f'Reload {news_source}')
            df = load_some_enc(par).reset_index()
            print(df.shape[0])
            if 'permno' not in df.columns:
                if news_source == NewsSource.EIGHT_PRESS:
                    temp = data.load_list_by_date_time_permno_type()[['permno','adate','form_id','cik']].rename(columns={'adate':'date'}).drop_duplicates()
                    temp = temp.loc[~temp[['date','permno']].duplicated(),:]
                    temp = temp.loc[~temp[['form_id','cik']].duplicated(),:]
                    df['id']=np.nan
                    df['alert']=np.nan
                    df = df.merge(temp,how='left')
                if news_source  == NewsSource.NEWS_REF:
                    df['date']=df['index'].apply(lambda x:str(x[2]).split('T')[0])
                    df['date'] = pd.to_datetime(df['date'],errors='coerce')
                    df['ticker']=df['index'].apply(lambda x:str(x[-1]))
                    df['id']=df['index'].apply(lambda x:str(x[0]))
                    df['alert']=df['index'].apply(lambda x:str(x[-2]))
                    temp = data.load_crsp_daily()[['date', 'ticker', 'permno']].dropna().drop_duplicates()
                    temp['date'] = pd.to_datetime(temp['date'], errors='coerce')
                    temp = temp.loc[~temp[['date', 'permno']].duplicated(), :]
                    temp = temp.loc[~temp[['date', 'ticker']].duplicated(), :]
                    df = df.merge(temp,how='left')
                if news_source  == NewsSource.NEWS_THIRD:
                    df['date']=df['timestamp'].apply(lambda x:str(x).split('T')[0])
                    df['date'] = pd.to_datetime(df['date'],errors='coerce')
                    temp = data.load_crsp_daily()[['date', 'ticker', 'permno']].dropna().drop_duplicates()
                    temp['date'] = pd.to_datetime(temp['date'],errors='coerce')
                    temp = temp.loc[~temp[['date', 'permno']].duplicated(), :]
                    temp = temp.loc[~temp[['date', 'ticker']].duplicated(), :]
                    df = df.merge(temp,how='left')
            df = df[['permno','date','id','alert']]
            df.to_pickle(par.get_vec_process_dir(merged_bow=True,index_permno_only=True))
            print(df.shape[0]) # checkign that the size is unchanged
            sources.append(df)
        else:
            sources.append(pd.read_pickle(par.get_vec_process_dir(merged_bow=True, index_permno_only=True)))
    return sources


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
    for f in ['third']:
        print('Start working on ',f)
        df = pd.read_pickle(start_dir + f+'.p')

        ind = df['audiences'].apply(lambda l: any([':PRN'.lower() in str(x).lower() for x in l]))
        ind_p = df['provider'].apply(lambda x: ':PRN'.lower() in x.lower())
        df['prn'] = ind | ind_p
        df[['id','prn']].to_pickle(data.p_dir + 'prn.p')


