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

def create_the_match_df_with_ref(item_number, load_wsj = False):
    if load_wsj:
        cos = data.load_main_cosine('wsj_one_per_stock')
        dest_dir = 'res/match_wsj/'
    else:
        cos = data.load_main_cosine()
        dest_dir = 'res/match/'
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
    cos = cos.loc[cos['items']==item_number,:]
    load_dir = data.p_news_tickers_related
    if load_wsj:
        df = data.load_wsj_one_per_tickers()
        crsp = data.load_crsp_daily()[['date', 'permno', 'ticker']].drop_duplicates()
        df = df.merge(crsp).rename(columns={'date': 'form_date'})
    else:
        df = pd.read_pickle(load_dir+'ref.p')
    # df = df.drop(columns='match')
    temp  = cos[['news_id','match_ratio','form_date','permno','value']].drop_duplicates().rename(columns={'news_id':'id','value':'cos_sim'})
    temp['match'] = True
    df = df.merge(temp,how='left')
    df['match'] = df['match'].fillna(False)
    df['alert'] = df['body'].apply(len)==0
    df = df.loc[df['match'], ['headline','body','match_ratio','permno','form_date','cos_sim']].sort_values('match_ratio',ascending=False)
    df['permno'] = df['permno'].astype(int)
    #
    rel = data.load_some_relevance_icf()
    rel = rel.loc[rel['items']==item_number,['adate','permno','form_id']].rename(columns={'adate':'form_date'})
    temp_c = pd.DataFrame()
    for f in tqdm.tqdm([x for x in os.listdir(data.p_eight_k_clean) if 'press' in x],'load txt of press release'):
        temp = pd.read_pickle(data.p_eight_k_clean+f)
        ind = temp['form_id'].isin(rel['form_id'])
        temp_c = pd.concat([temp_c,temp.loc[ind,:]],axis=0)
    rel = rel.merge(temp_c[['form_id','txt']])
    df = df.merge(rel)
    os.makedirs(dest_dir,exist_ok=True)
    df['txt'] = df['txt'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in str(x)]))
    df.sort_values('cos_sim', ascending=False).to_excel(f'{dest_dir}match_{str(item_number).replace(".", "")}.xlsx')
    # df.sort_values('cos_sim', ascending=False).to_csv(f'{dest_dir}match_{str(item_number).replace(".", "")}.csv')
    print('Saved',item_number, 'to',dest_dir,flush=True)



if __name__ == '__main__':
    par = Params()
    data = Data(par)
    min_obs_for_match = 50
    perc_above_for_match = 0.01
    for item_number in Constant.LIST_ITEMS_FULL:
        try:
            create_the_match_df_with_ref(item_number,True)
        except:
            print(f'Skipped {item_number}')







































