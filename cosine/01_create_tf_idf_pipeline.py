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
from data import load_some_enc

# Let's assume you have a function that gives you batches of documents
# Example of such a function:
def get_next_batch(text, batch_size):
    # This is just a placeholder, replace this with actual logic to fetch batches from disk or database
    for i in range(0, len(text), batch_size):
        yield text[i:i + batch_size]


if __name__ == '__main__':
    par = Params()
    data = Data(par)
    df = pd.DataFrame()
    par.enc.opt_model_type = OptModelType.BOW1
    save_dir = par.get_tf_idf_dir()

    documents_bow = []
    for news_source in [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD]:
        par.enc.news_source = news_source
        df = load_some_enc(par)
        documents_bow += df.values.tolist()
    del df
    gc.collect()

    # Assuming `documents_bow` is your list of BoW dictionaries for the documents
    documents_bow = [json.loads(bow_str[0]) for bow_str in tqdm.tqdm(documents_bow, 'decoding the bow')]


    # 1. Convert your BoW dictionaries to Gensim's format
    # This creates a mapping of word -> id, and will be used to convert your dictionaries
    dictionary = corpora.Dictionary()
    # for batch in tqdm.tqdm(get_next_batch(documents_bow, batch_size=1000),'Updated dictionary'):
    #     Update the dictionary
        # dictionary.add_documents([list(doc.keys()) for doc in batch])
    dictionary.add_documents([list(doc.keys()) for doc in tqdm.tqdm(documents_bow,'Add documents to dictionary')])

    # dictionary.filter_extremes(no_below=par.tfidf.no_below, no_above=par.tfidf.no_above, keep_n=par.tfidf.dict_size)
    gensim_bow = [dictionary.doc2bow(doc) for doc in tqdm.tqdm(documents_bow,'Building gensim doc')]
    # 2. Save your BoW representations
    outp = save_dir +f'corpus_{par.enc.opt_model_type.name}'
    MmCorpus.serialize(outp + '_bow.mm', gensim_bow, progress_cnt=10000, metadata=True)
    print('Saved McCorpus',flush=True)
    dictionary.save_as_text(outp + '_wordids.txt.bz2')
    print('Dictionary',flush=True)

    # # 3. Load the saved BoWs
    # mm = MmCorpus(outp + '_bow.mm')
    #
    # # 4. Compute and save the TF-IDF representations
    # # tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
    # # tfidf = TfidfModel(mm, id2word=dictionary, normalize=True, smartirs='ntc')
    # tfidf = TfidfModel(mm, id2word=dictionary, normalize=True, smartirs='ltc')
    #
    # tfidf.save(outp + '.tfidf_model')
    # MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)



