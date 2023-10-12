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



if __name__ == '__main__':
    par = Params()
    args = didi.parse()
    data = Data(par)
    df = pd.DataFrame()
    par.enc.opt_model_type = OptModelType.BOW1
    par.enc.news_source = NewsSource.NEWS_THIRD
    load_dest = par.get_cosine_dir(temp=True)
    save_dir = par.get_cosine_dir(temp=False)
    df = pd.DataFrame()
    print(f'Start Merging {len(os.listdir(load_dest))} permnos',flush=True)
    for f in tqdm.tqdm(os.listdir(load_dest),'Merging each permno'):
        try:
            df = pd.concat([df,pd.read_pickle(load_dest+f)],axis=0)
        except:
            print(f'bug {f}')

    df.to_pickle(save_dir+'df.p')
