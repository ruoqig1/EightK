import pandas as pd
import tensorflow as tf
import os
from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm

from parameters import *
from data import Data

par = Params()
data = Data(par)

ev = data.load_list_by_date_time_permno_type()


par.enc.news_source = NewsSource.EIGHT_LEGAL
for press_or_legal in ['press','legal']:
    res = pd.DataFrame()
    for year in  tqdm.tqdm(np.arange(2004, 2023),'merge'):
        df = pd.read_pickle(data.p_eight_k_clean + f'{press_or_legal}_{year}.p')
        df = df.loc[df['ran'],:]
        df = df.dropna()
        df['len'] = df['txt'].apply(lambda x: len(x.split(' ')))
        if press_or_legal == 'press':
            df=df[['cik','form_id','len']]
            # df.columns=df.columns =['cik','items','form_id','len']
        else:
            df=df[['cik','item','form_id','len']]
            df.columns=df.columns =['cik','items','form_id','len']
        res = pd.concat([res,df],axis=0)
    res['cik'] = res['cik'].astype(int)
    df = ev.merge(res)
    df =df.to_pickle(data.p_dir+f'ss_count_{press_or_legal}.p')

