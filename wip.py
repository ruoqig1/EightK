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

# e 1.02, 1.03, 2.01 to 2.06, 3.01 to 3.03, 4.01, 4.02, 5.01 item_number

if __name__ == '__main__':
    par = Params()
    data = Data(par)
    min_obs_for_match = 50
    perc_above_for_match = 0.01

    # df = pd.read_pickle('data/cosine/opt_model_typeOptModelType.BOW1news_sourceNewsSource.WSJ_ONE_PER_STOCKnb_chunks100save_chunks_size500chunk_to_run_id1/df.p')
    # df = pd.read_pickle('data/cosine/opt_model_typeOptModelType.BOW1news_sourceNewsSource.WSJ_ONE_PER_STOCKnb_chunks100save_chunks_size500chunk_to_run_id1/df.p')
    df = pd.read_pickle('data/cosine/opt_model_typeOptModelType.BOW1news_sourceNewsSource.WSJ_ONE_PER_STOCKnb_chunks100save_chunks_size500chunk_to_run_id1/df.p')
    df['news_prov'].unique()

    df.loc[df['news_prov']==3,'news_id']
    os.listdir('data/cosine/')
    df.loc[df['dist'].between(0,1),:].groupby('news_prov')['value'].mean()
    df.loc[df['dist'].between(-1,-1),:].groupby('news_prov')['value'].mean()

    for news_prov in np.sort(df['news_prov'].unique()):
        temp = df.loc[(df['news_prov']==news_prov)&(df['dist'].abs()<10),:]
        t= temp.groupby('dist')['value'].mean()
        # t=t/t.iloc[0]
        t.plot()
        plt.title(news_prov)
        plt.tight_layout()
        plt.show()


    # df.loc[(df['news_prov'] == 1), 'value'].hist(bins=100)
    # plt.show()
    #
    # df.loc[(df['news_prov'] == 3), 'value'].mean()