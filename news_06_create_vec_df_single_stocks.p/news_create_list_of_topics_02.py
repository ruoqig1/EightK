import os

import numpy as np
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


# Function to find the first non-zero column for each row
def find_first_non_zero(row, columns):
    for col in columns:
        if row[col] != 0:
            return col
    return None


if __name__ == "__main__":
    args = didi.parse()  # 27 variations

    data = Data(Params())
    reload = True

    save_dir = Constant.MAIN_DIR + 'res/ss/tickers_and_groups/'
    ss_dir = Constant.MAIN_DIR + 'res/ss/'
    os.makedirs(ss_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    final_dir = Constant.MAIN_DIR+'res/list_usage/'
    final_dir2 = Constant.MAIN_DIR+'res/list_usage2/'
    os.makedirs(final_dir2, exist_ok=True)
    min_obs = 1000
    nb_sample =10
    # 0 = refi, 1 = 3rdparty
    for refi_id, f in enumerate(os.listdir(final_dir)):
        start_dir = data.p_news_year if refi_id==0 else data.p_news_third_party_year
        df = pd.read_csv(final_dir+f,index_col=0)
        ind = df['tot']>=min_obs
        df =df.loc[ind,:]
        ind=df['born'].apply(len)!=1
        df = df.loc[ind,:].reset_index(drop=True)

        month=df.iloc[:,2:]
        df=df.iloc[:,:2]
        ((month>0)*1)
        df['FirstNonZeroColumn'] = month.apply(lambda row: find_first_non_zero(row, month.columns), axis=1)
        for j in range(nb_sample):
            df[j] = np.nan

        curent_ym = ''
        print(df.shape)
        for i in tqdm.tqdm(df.index,'Main loop'):
            m = month.loc[i,:]
            m = list(m.index[m>nb_sample])
            if curent_ym not in m:
                curent_ym = np.random.choice(m,1)[0]
                base = [x.split('_')[0] for x in os.listdir(start_dir)][0]
                news=pd.read_pickle(start_dir+f'{base}_{curent_ym[:-2]}-{curent_ym[-2:]}.p')

            born = df.loc[i,'born']
            for c in ['audiences','provider','subjects']:
                ind = news['subjects'].apply(lambda l: any([x==born for x in l]))
                if ind.sum()>nb_sample:
                    t=news.loc[ind,'headline'] + '\n \n' + news.loc[ind,'body']
                    sample = np.random.choice(t.values, nb_sample)
                    for j in range(nb_sample):
                        df.loc[i,j] = sample[j]
                    break
            # print(df.shape)

        df.to_csv(final_dir2+f'df_{refi_id}.csv')