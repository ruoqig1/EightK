import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *

def df_size(df):
    memory_in_gb = df.memory_usage(deep=True).sum() / 1e9
    return memory_in_gb

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    final = pd.DataFrame()

    start_dir = data.p_news_tickers_related
    crsp = data.load_crsp_daily()
    print('loaded')

    f = 'ref.p'
    df = pd.read_pickle(start_dir + f)

    df['ticker'].apply(len)==1
    df.mer
    # for f in tqdm.tqdm(os.listdir(start_dir),'Dropping and merging'):
    #     df = pd.read_pickle(start_dir+f)
    #     break

