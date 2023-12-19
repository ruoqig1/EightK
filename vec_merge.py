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
        if args.eight == 1:
            if args.legal == 1:
                print('args.ati', args.ati,flush=True)
                par = Params()
                par.enc.opt_model_type = OptModelType.OPT_13b
                par.enc.news_source = NewsSource.EIGHT_LEGAL
                vec = pd.DataFrame()
                for f in tqdm.tqdm(os.listdir(par.get_vec_process_dir()), 'merge the vectors'):
                    t = pd.read_pickle(par.get_vec_process_dir() + f).reset_index()
                    vec = pd.concat([vec, t], axis=0)
                vec = vec.rename(columns={'item': 'items'})
                data = Data(par)
                if args.ati == 1:
                    print('training with legalt_ati',flush=True)
                    par.enc.news_source = NewsSource.EIGHT_LEGAL_ATI
                    df = data.load_icf_ati_filter(training=False)
                    print(df.head(),flush=True)
                elif args.ati == 2:
                    par.enc.news_source = NewsSource.EIGHT_LEGAL_ATI_TRAIN
                    print('training with legalt_ati_TRAIN',flush=True)
                    df = data.load_icf_ati_filter(training=True)
                    print(df.head(),flush=True)
                else:
                    df = data.load_return_for_nlp_on_eightk()
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




'''
merge the vectors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 913/913 [01:44<00:00,  8.72it/s]
       cik             form_id items                                           vec_last       date  permno       mcap_e      avg_vol  news0       ret     abret
0  1173431  000095012310029463  2.04  [-0.21214074, -0.35219863, 0.17579144, -0.7179... 2010-03-29   84597  11206965.00  6258992.000    1.0  0.003701  0.001783
1  1174922  000134100410000422  7.01  [-0.16481644, 0.052500386, 0.049718592, -1.334... 2010-02-25   89533   7742988.50  2377907.000    1.0 -0.000534 -0.007568
2  1173489  000095012310006631  5.02  [0.32430503, -0.42260602, -0.07971897, -0.3082... 2010-01-29   89572    242912.70    85651.047    1.0 -0.000946  0.001788
3  1174922  000134100410000423  8.01  [-0.083781816, -0.7832653, 0.3230438, 0.050013... 2010-02-25   89533   7742988.50  2377907.000    1.0 -0.000534 -0.007568
4  1173489  000095012310030118  5.02  [-0.60325754, -0.45229945, 0.043728404, -0.247... 2010-03-30   89572    254677.67    79985.383    1.0 -0.011867 -0.012804
date
2004    14825
2005    48551
2006    48586
2007    47301
2008    44613
2009    40944
2010    40654
2011    40835
2012    40074
2013    39707
2014    40116
2015    41701
2016    41188
2017    41458
2018    40500
2019    40169
2020    46862
2021    44352
2022    40354

'''