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
    item_to_use = 5.02
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
    cos = cos.loc[cos['items']==item_to_use,:]

    load_dir = data.p_news_tickers_related
    df = pd.read_pickle(load_dir+'ref.p')
    # df = df.drop(columns='match')

    temp  = cos[['news_id','match_ratio','form_date','permno','value']].drop_duplicates().rename(columns={'news_id':'id','value':'cos_sim'})
    temp['match'] = True
    df = df.merge(temp,how='left')
    df['match'] = df['match'].fillna(False)
    df['alert'] = df['body'].apply(len)==0

    df.loc[df['match'] ,'body'].iloc[100]

    COL_KEEP = ['headline','body','match_ratio','permno','form_date','cos_sim']
    df = df.loc[df['match'], COL_KEEP].sort_values('match_ratio',ascending=False)
    df['permno'] = df['permno'].astype(int)




    df =  data.load_some_relevance_icf()
    df = df.loc[df['items'].isin(Constant.LIST_ITEMS_TO_USE),:]
    df['some_news'] = df['no_rel']==0
    df = df.groupby(['mcap_d','form_id'])['some_news'].mean().reset_index()

    t=df.groupby('mcap_d')['some_news'].mean()
    plt.bar(t.index,t.values)
    plt.xlabel('Decile Market Cap')
    plt.ylabel('Percentage of 8k covered by news')
    plt.grid()
    plt.tight_layout()
    plt.show()

    df