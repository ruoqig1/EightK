import os

import pandas as pd
import pyperclip
import tqdm

from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, PlotPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
from utils_local.zip import decompress_gz_file
import json
import glob
import itertools
import re
from utils_local.zip import decompress_gz_file, unzip_all
from utils_local.nlp_ticker import *
##### THIRD PARTY!!!


def main(args):
    for do_third_party in [0, 1]:
        print('#' * 50)
        print('do_third_party', do_third_party)
        print('#' * 50, flush=True)
        load_start = data.p_news_year if do_third_party == 0 else data.p_news_third_party_year
        # year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(load_start)]))  # 28 variations
        # year_list = np.arange(1996,2023,1)
        month_full_list = os.listdir(load_start)
        month_todo = [month_full_list[args.a]]
        print(args.a, type(args.a))
        # year_todo = year_list[args.a]

        for month in month_todo:
            month_nb = month.split('-')[1].split('.p')[0]
            year_todo = month.split('-')[0].split('_')[1]
            print('Start month', month_nb, flush=True)
            save_name = f'refi_{year_todo}-{month_nb}_' if do_third_party == 0 else f'third_party_{year_todo}-{month_nb}_'

            df = pd.read_pickle(load_start + f'{month}')
            df = df.reset_index(drop=True)

            # Index(['source', 'name', 'timestamp', 'id', 'body', 'mimeType', 'firstCreated', 'versionCreated', 'takeSequence', 'pubStatus', 'language', 'altId', 'headline', 'subjects', 'audiences', 'provider', 'instancesOf', 'urgency'],
            #       dtype='object')

            tickers = df['subjects'].apply(lambda x: [xx.replace('R:', '') for xx in x if 'R:' in xx])
            subjects = df['subjects'].apply(lambda x: [xx for xx in x if 'R:' not in xx])
            audiences = df['audiences'].apply(lambda x: [xx for xx in x if 'R:' not in xx])
            provider = df['provider'].apply(lambda x: [xx for xx in x if 'R:' not in xx])

            unique_tickers = list(np.unique(list(itertools.chain.from_iterable(tickers))))
            pd.Series(unique_tickers).to_pickle(save_dir + save_name + '_tickers.p')
            print('save to,', save_dir + save_name + '_tickers.p')

            if not Constant.IS_VM:
                res = pd.Series()
                for c in [provider, audiences, subjects]:
                    un = list(np.unique(list(itertools.chain.from_iterable(c))))
                    print('unique', len(un))
                    cnt_dict = {x: c.astype('str').str.contains(x).sum() for x in tqdm.tqdm(un)}
                    res = pd.concat([res, pd.Series(cnt_dict)], axis=0, ignore_index=False)
                res.to_pickle(save_dir + save_name + '_count.p')
                print('save to,', save_dir + save_name + '_count.p')


if __name__ == "__main__":
    args = didi.parse()  # 27 variations

    data = Data(Params())
    reload = True

    save_dir = Constant.MAIN_DIR + 'res/ss/tickers_and_groups/'
    ss_dir = Constant.MAIN_DIR + 'res/ss/'
    os.makedirs(ss_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if Constant.IS_VM:
        for i in tqdm.tqdm(range(324)):
            args.a = i
            main(args)
        s = pd.Series()
        for f in os.listdir(save_dir):
            t = pd.read_pickle(save_dir+f)
            s=pd.concat([s,t],axis=0)
        s.drop_duplicates().to_pickle(ss_dir+'ticker_all.p')

        ind = s.str.contains('.')
        s = s.loc[ind]
        ind = s.apply(lambda x: len(x.split('.'))) == 2
        s = s.loc[ind]

        ticker = s.apply(lambda x: x.split('.')[0])
        market = s.apply(lambda x: x.split('.')[1])

        df = s.reset_index().rename(columns={0: 'full'})
        df['ticker'] = ticker.values
        df['market'] = market.values
        t = df.groupby('ticker')['market'].count().sort_values(ascending=False)

        # drop the numbers ones
        ind = df['ticker'].apply(has_numbers)
        df = df.loc[~ind, :]

        # remove the case wher ethe lower case indicate type of shares
        ind = df['ticker'].apply(has_mixed_case)
        df.loc[ind, 'ticker'] = df.loc[ind, 'ticker'].apply(remove_lowercase)

        # remove the ipos
        for var in ['IPO-', 'IPO-']:
            ind = df['ticker'].str.contains('-IPO')
            df.loc[ind, 'ticker'] = df.loc[ind, 'ticker'].apply(lambda x: x.split(var)[0])

        # remove weird symbols
        ind = df['ticker'].apply(has_weird_symbols)
        df.loc[ind, 'ticker'] = df.loc[ind, 'ticker'].apply(remove_weird_symbols)

        tickers_in_news = df['ticker'].str.upper().drop_duplicates()

        tickers_in_crsp = data.load_crsp_daily()['ticker'].drop_duplicates()

        ticker_in_both = tickers_in_news[tickers_in_news.isin(tickers_in_crsp)]
        ticker_in_both.to_pickle(data.p_dir+'ticker_in_news_and_crsp.p')

    else:
        if int(args.a) > 0:
            main(args)
        else:
            df_refi = None
            df_third = None
            for f in [x for x in os.listdir(save_dir) if '_count' in x]:
                print(f)
                month = f.split('-')[1].split('__')[0]
                year = f.split('-')[0].split('_')[-1]
                t= pd.read_pickle(save_dir+f).reset_index().rename(columns={'index':'born',0:int(f'{year}{month}')})
                if 'refi' in f:
                    if df_refi is None:
                        df_refi = t
                    else:
                        df_refi=df_refi.merge(t,how='outer')
                else:
                    if df_third is None:
                        df_third = t
                    else:
                        df_third=df_third.merge(t,how='outer')

            final_dir = Constant.MAIN_DIR+'res/list_usage/'
            os.makedirs(final_dir,exist_ok=True)

            for i, df in enumerate([df_refi,df_third]):
                df =df.fillna(0.0)
                l= list(np.sort(df.columns[1:]))
                df['tot'] =  df[l].sum(1)
                df=df[['born','tot']+l]
                df=df.sort_values('tot')
                df.to_csv(final_dir+f'list_{i}.csv')
                print(f'Saving {i}')
