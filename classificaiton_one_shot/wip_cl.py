import pandas as pd
import tensorflow as tf
import os
from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm
from didipack import PandasPlus

from parameters import *
from data import Data
from matplotlib import pyplot as plt
from didipack import OlsPLus

# e 1.02, 1.03, 2.01 to 2.06, 3.01 to 3.03, 4.01, 4.02, 5.01 5.02

if __name__ == '__main__':
    par = Params()
    data = Data(par)
    min_obs_for_match = 50
    perc_above_for_match = 0.01
    cos = data.load_main_cosine()
    cos = cos.loc[cos['news_prov']==1,:]
    ind = cos['dist'].between(-1,1)
    cos.loc[ind,'value'].mean()
    cos = cos.merge(cos.loc[~ind,:].groupby('permno')['value'].aggregate(['count','mean']).reset_index())
    cos['match_ratio'] = cos['value']/cos['mean']
    cos['match'] = (cos['count']>=min_obs_for_match) & ((cos['value']/cos['mean']-1)>=perc_above_for_match)

    cos = cos.loc[cos['dist'].between(0,1) & cos['match'],:]
    cos['permno'] = cos['permno'].astype(int)
    items = data.load_some_relevance_icf()[['items','permno','adate']].rename(columns={'adate':'form_date'})
    cos = cos.merge(items)
    cos = cos.loc[cos['items']==5.02,:]

    load_dir = data.p_news_tickers_related
    df = pd.read_pickle(load_dir+'ref.p')
    # df = df.drop(columns='match')

    temp  = cos[['news_id','match_ratio','form_date','permno','value']].drop_duplicates().rename(columns={'news_id':'id','value':'cos_sim'})
    temp['match'] = True
    df = df.merge(temp,how='left')
    df['match'] = df['match'].fillna(False)
    df['alert'] = df['body'].apply(len)==0

    df.loc[df['match'] ,'body'].iloc[100]

    df = df.loc[df['match'], ['headline','body','match_ratio','permno','form_date','cos_sim']].sort_values('match_ratio',ascending=False)
    df['permno'] = df['permno'].astype(int)
    #

    rel = data.load_some_relevance_icf()
    rel = rel.loc[rel['items']==5.02,['adate','permno','form_id']].rename(columns={'adate':'form_date'})

    temp_c = pd.DataFrame()
    for f in tqdm.tqdm([x for x in os.listdir(data.p_eight_k_clean) if 'press' in x],'load txt of press release'):
        temp = pd.read_pickle(data.p_eight_k_clean+f)
        ind = temp['form_id'].isin(rel['form_id'])
        temp_c = pd.concat([temp_c,temp.loc[ind,:]],axis=0)
    rel = rel.merge(temp_c[['form_id','txt']])

    df = df.merge(rel)
    df.sort_values('cos_sim',ascending=False).to_excel('res/ceo_match.xlsx')


