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
from utils_local.llm import EncodingModel
from utils_local.zip import decompress_gz_file, unzip_all
from utils_local.vec_functions import vectorise_in_batch


def load_all_enc(par: Params):
    load_dir = par.get_vec_process_dir()
    df = pd.DataFrame()
    msg = f'Merging enc of {par.enc.opt_model_type.name}, {par.enc.news_source.name}'
    for f in tqdm.tqdm(os.listdir(load_dir), msg):
        print(f'loading {f}')
        t = pd.read_pickle(load_dir + f)
        print(t.shape,flush=True)
        df = pd.concat([df, t], axis=0)
    return df


if __name__ == "__main__":
    args = didi.parse()  # --legal=0/1 --eight=0/1 |  19 variations for legal eight
    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    if args.bow == 0:
        if args.eightsl1:
            if args.legal == 1:
                par = Params()
                par.enc.opt_model_type = OptModelType.OPT_13b
                par.enc.news_source = NewsSource.EIGHT_LEGAL
                data = Data(par)
                df = data.load_return_for_nlp_on_eightk()
                ev = data.load_list_by_date_time_permno_type()
                vec = pd.DataFrame()
                for f in tqdm.tqdm(os.listdir(par.get_vec_process_dir()), 'merge the vectors'):
                    t = pd.read_pickle(par.get_vec_process_dir() + f).reset_index()
                    vec = pd.concat([vec, t], axis=0)
                vec = vec.rename(columns={'item': 'items'})
                df = df.drop_duplicates()
                df = vec.merge(df)
                print(df.head(), flush=True)
                print(df.groupby(df['date'].dt.year)['cik'].count(), flush=True)
        if args.news_on_eight == 1:
            # merging
            par = Params()
            par.enc.opt_model_type = OptModelType.OPT_13b
            par.enc.news_source = NewsSource.NEWS_REF
            data = Data(par)

            ev = data.load_list_by_date_time_permno_type()
            ev['date'] = pd.to_datetime(ev['adate'])
            vec = pd.DataFrame()
            todo = [x for x in os.listdir(par.get_vec_process_dir()) if (int(x.split('_')[0])>=2004)]
            for f in tqdm.tqdm(todo, 'merge the vectors'):
                t = pd.read_pickle(par.get_vec_process_dir() + f).reset_index()
                t['date'] = t['timestamp'].apply(lambda x: x.split('T')[0])
                t['date'] = pd.to_datetime(t['date'], errors='coerce')
                # now w reduce to the dates that have at least one eightk!
                t = t.merge(ev[['date','ticker','permno']].drop_duplicates())
                if t.shape[0]>0:
                    vec = pd.concat([vec, t], axis=0)
            # vec = vec.rename(columns={'item': 'items'})
            df = data.load_return_for_nlp_on_eightk()
            df = df[['permno','date','news0','ret','abret']].drop_duplicates()
            df = vec.merge(df)
            print(df.head(), flush=True)
            print(df.groupby(df['date'].dt.year)['permno'].count(), flush=True)
            par.enc.news_source = NewsSource.NEWS_REF_ON_EIGHT_K
            print('Finishing news ref on eight k')

        # saving the 13b version
        save_dir = par.get_training_dir()
        print('Saving in',save_dir,flush=True)
        x = np.vstack([x.reshape(1, -1) for x in df['vec_last'].values])
        np.save(save_dir + 'x', x)
        df.drop(columns='vec_last').to_pickle(save_dir + 'main_df.p')
    else:
        # HERE WE ARE IN THE MERGE BOW
        print('Start merge bow')
        par = Params()
        data = Data(par)
        par.enc.opt_model_type = OptModelType.BOW1
        for news_source in [NewsSource.WSJ_ONE_PER_STOCK]:
        # for news_source in [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF, NewsSource.NEWS_THIRD, NewsSource.WSJ_ONE_PER_STOCK]:
            par.enc.news_source = news_source
            df = load_all_enc(par)
            save_dest = par.get_vec_process_dir(merged_bow=True)
            df.to_pickle(save_dest)
            print('Saved to', save_dest)


