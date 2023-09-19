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
from utils_local.llm import OPTModel
from utils_local.zip import decompress_gz_file, unzip_all
from utils_local.vec_functions import vectorise_in_batch


def load_and_process_eight_k_legal(par,args):
    save_size = 1000
    # load and pre process data (code specific)
    data = Data(par)
    # those are the events for which we have a match in crsp
    ev = data.load_list_by_date_time_permno_type()
    ev= ev[['cik','form_id']].drop_duplicates()
    year = np.arange(2004,2023)[args.a]
    # load the raw text of the year
    df=pd.read_pickle(data.p_eight_k_clean+f'legal_{year}.p')
    #make the dtypes match for the merge
    df['cik']=df['cik'].astype(int)
    sh = df.shape[0]
    # keep only the ones with a match
    df = df.merge(ev,how='inner')
    print('Droped',df.shape[0]/sh)
    print('Left',df.shape[0])

    df['item'] = pd.to_numeric(df['item'],errors='coerce')
    ind = df['item'].isin(Constant.LIST_ITEMS_TO_USE)
    df = df.loc[ind,:]
    df=df.drop_duplicates()

    # remove the text that have been extracted from multiple sources
    id_col = ['cik','form_id','item']
    df['len']=df['txt'].apply(lambda x:len(str(x)))
    ind = df.groupby(id_col)['len'].transform('max')==df['len']
    df = df.loc[ind,:].reset_index(drop=True)
    #remove the duplicates that have same lengths of text arbitrairly
    while df[id_col].duplicated().sum()!=0:
        ind = ~df[id_col].duplicated()
        df = df.loc[ind,:].reset_index(drop=True)


    return id_col,save_size,batch_size,year,df



if __name__ == "__main__":
    args = didi.parse() # --legal=0/1 --eight=0/1 |  19 variations for legal eight
    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    if socket.gethostname() == '3330L-214940-M':
        # (local debug)
        par.enc.opt_model_type = OptModelType.OPT_125m
        batch_size = 2

        par.enc.news_source = NewsSource.EIGHT_LEGAL
        id_col, save_size, batch_size, year, df = load_and_process_eight_k_legal(par, args)
        save_size=10

        # launch the vectorisation
        vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year)
    else:
        par.enc.opt_model_type = OptModelType.OPT_13b
        batch_size = 2

    if args.eight==1:
        if args.legal==1:
            par.enc.news_source = NewsSource.EIGHT_LEGAL
            id_col, save_size, batch_size, year, df = load_and_process_eight_k_legal(par,args)

            # launch the vectorisation
            vectorise_in_batch(id_col =id_col, df=df, save_size=save_size, batch_size=batch_size, par =par, year=year)

