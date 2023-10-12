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

from gensim import similarities


if __name__ == '__main__':
    par = Params()
    data = Data(par)
    df = pd.DataFrame()
    par.enc.opt_model_type = OptModelType.BOW1
    save_dir = par.get_tf_idf_dir()

    outp = save_dir +f'corpus_{par.enc.opt_model_type.name}'

    # # 3. Load the saved BoWs
    mm = MmCorpus(outp + '_bow.mm')
    dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')
    print('Loaded/setup')
    # 4. Compute and save the TF-IDF representations
    # tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    # tfidf = TfidfModel(mm, id2word=dictionary, normalize=True, smartirs='ntc')
    tfidf = TfidfModel(mm, id2word=dictionary, normalize=True, smartirs='ltc')

    tfidf.save(outp + '.tfidf_model')
    MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)
    print(f'save model to {outp}.tfidf_model')
#


