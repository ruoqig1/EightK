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
import re
import json
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)


def find_all_instance_of_subject(subj, res):
    try:
        subj = str(subj)
        ind=res['subjects'].apply(lambda x: any([subj in xx for xx in x]))
        ind = ind.sum()
    except:
        ind = np.nan
    return ind

def find_tickers_in_markup(text):
    return re.findall(r'<(.*?)>', text)

def find_tickers_in_square_brackets(text):
    return re.findall(r'\[(.*?)\]', text)
def get_unique_tickers(list_of_list):
    # Concatenate all the lists
    concatenated_tickers = list(itertools.chain.from_iterable(list_of_list))
    t=np.unique(concatenated_tickers)
    return list(t)

if __name__ == "__main__":
    args = didi.parse()

    print(args.para)
    if args.para ==1:
        pandarallel.initialize(nb_workers=8, progress_bar=True)

    par = Params()
    data = Data(par)
    all_items = data.load_all_item_list_in_refinitiv()
    unique_body = []
    unique_headline = []
    # for year in [1999,2006,2022]:
    # for year in range(1996,2023):
    # all the items
    df_items = data.load_all_item_list_in_refinitiv()
    years_todo = list(np.arange(1996,2023,1))
    # years_todo = [2006]
    df_sample_to_save = pd.DataFrame()

    for year in years_todo:
        print('Start working on year',year)
        save_dir = 'temp/luci/'
        os.makedirs(save_dir,exist_ok=True)

        df = pd.read_pickle(data.p_news_year+f'ref_{year}.p').reset_index(drop=True)

        ticker_headline = df['headline'].apply(find_tickers_in_markup)
        ticker_body = df['body'].apply(find_tickers_in_markup)
        a=get_unique_tickers(ticker_headline)
        b=get_unique_tickers(ticker_body)
        unique_headline = unique_headline + a
        unique_body = unique_body + b

        res=df[['body','headline','subjects','instancesOf']].copy()
        res['ticker_headline'] = ticker_headline
        res['ticker_body'] = ticker_body
        ind = (res['ticker_headline'].apply(len)>0) | (res['ticker_body'].apply(len)>0)
        # res.loc[ind,:].to_csv(save_dir+f'news_{year}.csv')
        # pd.Series(a).to_csv(save_dir+f'ticker_in_headline_{year}.csv')
        # pd.Series(b).to_csv(save_dir+f'ticker_in_body_{year}.csv')
        print('start the costly apply',flush=True)

        if args.para==1:
            # Use parallel_apply instead of the regular apply or progress_apply
            df_items['year'] = df_items['qode'].parallel_apply(find_all_instance_of_subject, res=res)
        else:
            tqdm.tqdm.pandas()
            df_items['year'] = df_items['qode'].progress_apply(find_all_instance_of_subject, res=res)
        # df_items[year]=df_items['qode'].apply(find_all_instance_of_subject,res=res)
        print('done',flush=True)
        res['year'] = year
        nb_sample_per_year = int(1e5)
        df_sample_to_save = pd.concat([df_sample_to_save,res.sample(nb_sample_per_year)],axis=0)

    a=list(np.unique(unique_headline))
    b=list(np.unique(unique_body))

    pd.Series(a).to_csv(save_dir + f'ticker_in_headline_all.csv')
    pd.Series(b).to_csv(save_dir + f'ticker_in_body_all.csv')
    df_sample_to_save.to_csv(save_dir + f'sample_news.csv')

    df_items['tot']=df_items[years_todo].sum(1)
    df_items.to_csv(save_dir + f'items_with_counts.csv')


