import gc

import pandas as pd
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from nltk.tokenize import word_tokenize
from parameters import *
from data import Data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import json
import joblib
from gensim import corpora, models
import logging
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
import didipack as didi
from gensim import similarities
from data import load_some_enc
from gensim import corpora, models, similarities
from gensim.corpora import MmCorpus
from gensim.models import TfidfModel

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
                    df['alert']=df['index'].apply(lambda x:str(x[-2])) # THIS SHIT IS SAVED IN BOOL! FUCK ME< FUCK LIFE< FUCK EVERYTHING!!
                    #correcting the lovely bug :) python lack of data defintion cuts boths ways andI hate life now :D
                    df['alert'] = df['alert']=='True'
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

if __name__ == '__main__':
    par = Params()
    args = didi.parse()
    data = Data(par)
    df = pd.DataFrame()
    par.enc.opt_model_type = OptModelType.BOW1
    save_dir = par.get_tf_idf_dir()
    days_kept = 60

    index_list = load_all_index(par,reload=True)
    for i in range(len(index_list)):
        print(np.sort(pd.to_datetime(index_list[i]['date']).dt.year.unique()))

    # Load the previously saved TF-IDF model and the transformed corpus
    outp = save_dir + f'corpus_{par.enc.opt_model_type.name}'
    tfidf = TfidfModel.load(outp + '.tfidf_model')
    corpus_tfidf = MmCorpus(outp + '_tfidf.mm')

    print(index_list[0].shape[0] + index_list[1].shape[0] + index_list[2].shape[0] - len(corpus_tfidf))

    permno_todo = index_list[0]['permno'].unique()
    temp_l = []
    for permno in permno_todo:
        save_dest = par.get_cosine_dir(temp=True) + f'{permno}.p'
        if not os.path.exists(save_dest):
            temp_l.append(permno)
    permno_todo = np.array(temp_l)


    permno_todo = np.array_split(permno_todo,3)[args.a]
    for permno in tqdm.tqdm(permno_todo,'loop through permno'):
        save_dest = par.get_cosine_dir(temp=True)+f'{permno}.p'
        if not os.path.exists(save_dest):
            tot_i = 0
            index_to_do = []
            dates = pd.DataFrame()
            for i in range(len(index_list)):
                df = index_list[i].reset_index(drop=True)
                df['permno'] = df['permno'].fillna(-1).astype(int)
                df['alert'] = df['alert'].fillna(False)
                ind = (df['permno']==permno) & (df['alert']==False)
                index_to_do.append(np.array(df.loc[ind,:].index)+tot_i)
                tp = df.loc[ind, ['date','id']]
                tp['type'] = i
                dates = pd.concat([dates,tp],axis=0)
                tot_i+= df.shape[0]
            index_to_do = np.concatenate(index_to_do)
            print('len index', len(index_to_do))
            if len(index_to_do)>0:
                subset_corpus_tfidf = [corpus_tfidf[i] for i in index_to_do]
                index = similarities.MatrixSimilarity(subset_corpus_tfidf)
                df = np.array(index)
                print('start melting')
                df = pd.DataFrame(df, columns=dates, index=dates).melt(ignore_index=False)
                df=df.reset_index()
                ind = (df['index'].apply(lambda x: x[-1]) == 0) & (df['variable'].apply(lambda x: x[-1]) > 0)
                df = df.loc[ind, :]
                df=df.rename(columns={'index': 'form', 'variable': 'news'})
                df['form_date'] = pd.to_datetime(df['form'].apply(lambda x: x[0]))
                df['news_date'] = pd.to_datetime(df['news'].apply(lambda x: x[0]))
                df['dist']= (df['form_date']-df['news_date']).dt.days
                ind = df['dist'].abs()<=days_kept
                df['news_id'] = df['news'].apply(lambda x: x[1])
                df['news_prov'] = df['news'].apply(lambda x: x[2])
                df = df.loc[ind,['form_date', 'news_date', 'dist', 'news_prov', 'news_id','value']]
                df['permno'] = permno
                df.to_pickle(save_dest)
                del df,index,subset_corpus_tfidf
        else:
            print(save_dest,'ALREADY DONE',flush=True)




