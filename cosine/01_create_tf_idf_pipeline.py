import gc
import json
import tqdm
from gensim import corpora
from gensim.corpora import MmCorpus
from data import Data
from data import load_some_enc
from parameters import *
from utils_local.general import get_news_source_to_do_for_tfidf
from experiments_params import get_params_for_tfidf
# Let's assume you have a function that gives you batches of documents
# Example of such a function:
def get_next_batch(text, batch_size):
    # This is just a placeholder, replace this with actual logic to fetch batches from disk or database
    for i in range(0, len(text), batch_size):
        yield text[i:i + batch_size]


if __name__ == '__main__':
    par = get_params_for_tfidf()
    data = Data(par)
    df = pd.DataFrame()
    save_dir = par.get_tf_idf_dir()

    documents_bow = []

    news_source_todo = get_news_source_to_do_for_tfidf(par)

    for news_source in news_source_todo:
    # for news_source in [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD]:
        par.enc.news_source = news_source
        df = load_some_enc(par)
        print(news_source,flush=True)
        print(df,flush=True)
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
    # filter extreme values
    if par.tfidf.do_some_filtering:
        dictionary.filter_extremes(no_below=par.tfidf.no_below, no_above=par.tfidf.no_above)

    gensim_bow = [dictionary.doc2bow(doc) for doc in tqdm.tqdm(documents_bow,'Building gensim doc')]
    # 2. Save your BoW representations
    outp = save_dir +f'corpus_{par.enc.opt_model_type.name}'

    MmCorpus.serialize(outp + '_bow.mm', gensim_bow, progress_cnt=10000, metadata=True)
    print('Saved McCorpus',flush=True)
    dictionary.save_as_text(outp + '_wordids.txt.bz2')
    print('Dictionary',flush=True)
    par.save(save_dir=save_dir)





# old one 5ab66cb7810d1c6f0adca831144a85aa1b45934210a4a8640a9913c567ebac2e