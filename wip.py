import gc

import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from itertools import chain

if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    final = pd.DataFrame()


    start_dir = data.p_news_tickers_related
    save_dir = data.p_to_vec_main_dir+'/single_stock_news_to_vec/'
    os.makedirs(save_dir,exist_ok=True)

    # f = 'sample'
    M = []
    for f in ['ref', 'third']:
        df = pd.read_pickle(start_dir + f + '.p')[['subjects']]
        gc.collect()
        tickers = df['subjects'].apply(lambda x: [xx.replace('R:', '') for xx in x if 'R:' in xx])

        markets = tickers.apply(lambda x: [get_market(xx) for xx in x])
        m = list(chain.from_iterable(list(markets.values)))
        m = [str(x).upper() for x in m]
        M.append(m)
    markets = pd.DataFrame(M).reset_index().groupby(0).count().to_csv(Constant.MAIN_DIR+'res/list_market.csv')


