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

##### THIRD PARTY!!!

if __name__ == "__main__":
    args = didi.parse() #28 variations

    par = Params()
    data = Data(par)

    if Constant.IS_VM:
        start_dir = '/mnt/layline/refinitiv/News/RTRS/Monthly/'
        unzip_dir = '/mnt/layline/project/eightk/NewsUnzip/RTRS/Monthly/'
    else:
        start_dir = 'data/refinitiv/News/3PTY/Monthly/'
        unzip_dir = 'data/refinitiv/NewsUnzip/3PTY/Monthly/'

    years_list = [int(x) for x in os.listdir(start_dir) if x.isdigit()]
    years_todo = years_list[int(args.a)]

    zip_dir = start_dir+f'{years_todo}/JSON/'
    unzip_dir = unzip_dir+f'{years_todo}/JSON/'
    os.makedirs(unzip_dir,exist_ok=True)

    # start by unzipping all the files for the year
    unzip_all(zip_dir,unzip_dir,False)

    df = pd.DataFrame()
    # for each file in the unzip fil, merge it all into a pd dataframe
    for f in tqdm.tqdm(os.listdir(unzip_dir),'Loop through month for merge'):
        # Replace 'your_file.json' with the path to your actual file
        with open(unzip_dir+f, 'r') as json_f:
            raw_news = json.load(json_f)
        dt = pd.DataFrame([x['data'] for x in raw_news['Items']])
        time = pd.DataFrame([x['timestamps'][0] for x in raw_news['Items']])
        dt = pd.concat([time,dt],axis=1)
        df = pd.concat([df,dt],axis=0)

        raw_news.keys()
        raw_news['MajorVersion']
        raw_news['MinorVersion']


    # remove a few usless news at this early stage

    # remove all the non english ones
    ind =df['language'] == 'en'
    print('English news are',ind.mean(), 'percent fo the news')
    df =df.loc[ind,:]
    ind_alert = df['body'] == ''
    print('Percentage of allerts',ind_alert.mean())

    # remve non alert with less than 100 characters, and detailed reports with more than 100,000 characters.
    nb_char = df['body'].apply(len)
    ind = (nb_char==0) | ((nb_char>=100) & (nb_char<=1e5))
    print('Percentage of news with ok length of char',ind.mean())
    df =df.loc[ind,:]

    save_to = data.p_news_year+f'ref_{years_todo}.p'
    df.to_pickle(save_to)
    print(f'Save it all to {save_to}',flush=True)


    # finally remove the unzip file
    [os.remove(f) for f in glob.glob(f"{unzip_dir}/*.txt")]

    print('All done',flush=True)


    #
    # ticker_headline = df['headline'].apply(find_tickers_in_markup)
    # ticker_body = df['body'].apply(find_tickers_in_markup)
    # def get_unique_tickers(list_of_list):
    #     # Concatenate all the lists
    #     concatenated_tickers = list(itertools.chain.from_iterable(list_of_list))
    #     t=np.unique(concatenated_tickers)
    #     return list(t)
    #
    # a=get_unique_tickers(ticker_headline)
    # b=get_unique_tickers(ticker_body)
    #
    # len(a)
    # len(b)
    # pyperclip.copy(df.loc[1974,'body'])
    # pyperclip.copy(df.loc[1688,'headline'])
