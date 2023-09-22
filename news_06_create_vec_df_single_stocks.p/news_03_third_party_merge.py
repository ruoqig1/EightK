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
        start_dir = '/mnt/layline/refinitiv/News/3PTY/Monthly/'
        unzip_dir = '/mnt/layline/project/eightk/NewsUnzip/3PTY/Monthly/'
    else:
        start_dir = 'data/refinitiv/News/3PTY/Monthly/'
        unzip_dir = 'data/refinitiv/NewsUnzip/3PTY/Monthly/'

    print('load from',start_dir,flush=True)

    years_list = [int(x) for x in os.listdir(start_dir) if x.isdigit()]
    years_list = list(np.sort(years_list))
    print('Year list',years_list,flush=True)
    print('len',len(years_list),flush=True)
    years_todo = years_list[int(args.a)]
    print('Year todo',years_todo,flush=True)

    zip_dir = start_dir+f'{years_todo}/JSON/'
    unzip_dir = unzip_dir+f'{years_todo}/JSON/'
    os.makedirs(unzip_dir,exist_ok=True)

    # start by unzipping all the files for the year
    unzip_all(zip_dir,unzip_dir,False)

    # for each file in the unzip fil, merge it all into a pd dataframe
    for f in tqdm.tqdm(os.listdir(unzip_dir),'Loop through month for merge'):
        # get the month
        month_nb = f.split(f'{years_todo}-')[1].split('.')[0]
        print(f'Start month {month_nb}')
        # Replace 'your_file.json' with the path to your actual file
        with open(unzip_dir+f, 'r') as json_f:
            raw_news = json.load(json_f)
        df = pd.DataFrame([x['data'] for x in raw_news['Items']])
        time = pd.DataFrame([x['timestamps'][0] for x in raw_news['Items']])
        df = pd.concat([time,df],axis=1)

        # remove all the non english ones
        ind =df['language'] == 'en'
        print('English news are',ind.mean(), 'percent fo the news')
        df =df.loc[ind,:]

        # remve non alert with less than 100 characters, and detailed reports with more than 100,000 characters.
        nb_char = df['body'].apply(len)
        ind = (nb_char==0) | ((nb_char>=100) & (nb_char<=1e5))
        print('Percentage of news with ok length of char',ind.mean())
        df =df.loc[ind,:]

        save_to = data.p_news_third_party_year+f'ref_{years_todo}-{month_nb}.p'
        df.to_pickle(save_to)
        print(f'Save it all to {save_to}',flush=True)


    # finally remove the unzip file
    # [os.remove(f) for f in glob.glob(f"{unzip_dir}/*.txt")]

    print('All done',flush=True)


