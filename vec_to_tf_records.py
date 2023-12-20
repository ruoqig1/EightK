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
from vec_merge import load_all_enc
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import pandas as pd


# Your data here (sample provided for reference)
# df = ...

def serialize_news_link_to_8k(row):
    # Extract values
    vec_last = row['vec_last']
    id_val = row['id']
    index_val = row['index']
    timestamp_val = row['timestamp']
    alert_val = int(row['alert'])  # Convert boolean to int for storage
    ticker_val = row['ticker']
    date_val = str(row['date'])
    permno_val = row['permno']
    news0_val = row['news0']
    ret_val = row['ret']
    abret_val = row['abret']
    m_cosine = row['m_cosine']
    cosine = row['cosine']
    prn = int(row['prn'])
    reuters = int(row['reuters'])
    # Create features
    feature = {
        'vec_last': tf.train.Feature(float_list=tf.train.FloatList(value=vec_last)),
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_val.encode()])),
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index_val])),
        'timestamp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[timestamp_val.encode()])),
        'alert': tf.train.Feature(int64_list=tf.train.Int64List(value=[alert_val])),
        'ticker': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ticker_val.encode()])),
        'date': tf.train.Feature(bytes_list=tf.train.BytesList(value=[date_val.encode()])),
        'permno': tf.train.Feature(int64_list=tf.train.Int64List(value=[permno_val])),
        'news0': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(news0_val)])),
        'ret': tf.train.Feature(float_list=tf.train.FloatList(value=[ret_val])),
        'abret': tf.train.Feature(float_list=tf.train.FloatList(value=[abret_val])),
        'cosine': tf.train.Feature(float_list=tf.train.FloatList(value=[cosine])),
        'm_cosine': tf.train.Feature(float_list=tf.train.FloatList(value=[m_cosine])),
        'prn': tf.train.Feature(int64_list=tf.train.Int64List(value=[prn])),
        'reuters': tf.train.Feature(int64_list=tf.train.Int64List(value=[reuters]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_news_link_all_single_news(row):
    # Extract values
    vec_last = row['vec_last']
    id_val = row['id']
    index_val = row['index']
    timestamp_val = row['timestamp']
    alert_val = int(row['alert'])  # Convert boolean to int for storage
    ticker_val = row['ticker']
    date_val = str(row['date'])
    permno_val = row['permno']
    ret_val = row['ret']
    ret_m = row['ret_m']
    reuters = int(row['reuters'])
    encoded_with_mean = int(row['vec_mean'])
    # Create features
    feature = {
        'vec': tf.train.Feature(float_list=tf.train.FloatList(value=vec_last)),
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_val.encode()])),
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index_val])),
        'timestamp': tf.train.Feature(bytes_list=tf.train.BytesList(value=[timestamp_val.encode()])),
        'alert': tf.train.Feature(int64_list=tf.train.Int64List(value=[alert_val])),
        'ticker': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ticker_val.encode()])),
        'date': tf.train.Feature(bytes_list=tf.train.BytesList(value=[date_val.encode()])),
        'permno': tf.train.Feature(int64_list=tf.train.Int64List(value=[permno_val])),
        'ret': tf.train.Feature(float_list=tf.train.FloatList(value=[ret_val])),
        'ret_m': tf.train.Feature(float_list=tf.train.FloatList(value=[ret_m])),
        'reuters': tf.train.Feature(int64_list=tf.train.Int64List(value=[reuters])),
        'encoded_with_mean': tf.train.Feature(int64_list=tf.train.Int64List(value=[encoded_with_mean])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
# Serialize DataFrame and write to TFRecord


if __name__ == "__main__":
    args = didi.parse()  # --legal=0/1 --eight=0/1 |  19 variations for legal eight
    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    if args.news_on_eight == 1:
        # merging
        year_todo = np.arange(2004, 2023, 1)[args.a]  # len 19
        print(f'Start running {year_todo}')
        par = Params()
        par.enc.opt_model_type = OptModelType.OPT_13b
        par.enc.news_source = NewsSource.NEWS_REF
        to_process_dir = par.get_vec_process_dir()
        par.enc.news_source = NewsSource.NEWS_REF_ON_EIGHT_K
        par.train.use_tf_models = True
        save_dir = par.get_training_dir()

        data = Data(par)
        cos = data.load_main_cosine()
        ev = data.load_list_by_date_time_permno_type()
        ev['date'] = pd.to_datetime(ev['adate'])
        df = data.load_return_for_nlp_on_eightk(False)
        cos_df = data.load_complement_id_for_tfidf_records(False).rename(columns={'news_id':'id'})
        prn_df = data.load_prn()

        df = df[['permno', 'date', 'news0', 'ret', 'abret']].drop_duplicates()
        vec = pd.DataFrame()
        todo = [x for x in os.listdir(to_process_dir) if (int(x.split('_')[0]) == year_todo)]
        for f in tqdm.tqdm(todo, 'merge the vectors'):
            vec = pd.read_pickle(to_process_dir + f).reset_index()
            vec['date'] = vec['timestamp'].apply(lambda x: x.split('T')[0])
            vec['date'] = pd.to_datetime(vec['date'], errors='coerce')
            # now w reduce to the dates that have at least one eightk!
            vec = vec.merge(ev[['date', 'ticker', 'permno']].drop_duplicates())

            save_dest = save_dir + f.replace('.p', '_data.tfrecord')
            if vec.shape[0] > 0:
                vec = vec.copy()
                vec = vec.merge(df)
                vec = vec.merge(cos_df,how='left')
                vec = vec.merge(prn_df,how='left')
                vec['reuters'] = pd.isna(vec['prn'])*1
                vec['prn']=vec['prn'].fillna(0.0)
                vec['cosine']=vec['cosine'].fillna(-1.0)
                vec['m_cosine']=vec['m_cosine'].fillna(100.0)
                with tf.io.TFRecordWriter(save_dest) as writer:
                    for _, row in vec.iterrows():
                        example = serialize_news_link_to_8k(row)
                        writer.write(example)

    if args.news_single == 1:
        # merging
        year_todo = np.arange(1996, 2023, 1)[args.a]  # len 27

        par = Params()
        par.enc.framework = Framework.TENSORFLOW
        if args.small ==1:
            par.enc.opt_model_type = OptModelType.OPT_125m
        else:
            par.enc.opt_model_type = OptModelType.OPT_13b
        to_process_dir_list = []
        source_list = ['third','ref']
        for source in [NewsSource.NEWS_THIRD, NewsSource.NEWS_REF]:
            par.enc.news_source = source
            to_process_dir_list.append(par.get_vec_process_dir())

        # set the destionation source now
        par.enc.news_source = NewsSource.NEWS_SINGLE
        # now we get the dir to save the processed tf records.
        par.train.use_tf_models = True
        save_dir = par.get_training_dir()
        print(f'Start running for all single news {year_todo}')
        # python3 vec_to_tf_records.py 0 --news_on_eight=0 --news_single=1 --small=1

        data = Data(par)
        df = data.load_crsp_daily()
        df['ret'] = pd.to_numeric(df['ret'],errors='coerce')
        df= df.sort_values(['permno','ret']).reset_index(drop=True)
        df['ret_m'] = (df.groupby('permno')['ret'].shift(1) + df.groupby('permno')['ret'].shift(-1) + df['ret'])
        df = df[['permno', 'date', 'ticker', 'ret','ret_m']].drop_duplicates().sort_values(['permno','date'])
        df =df.dropna(subset=['ticker','date','ret_m'])
        df = df.drop_duplicates()
        ind = ~df[['date','ticker']].duplicated()
        df = df.loc[ind,:].reset_index(drop=True)

        vec = pd.DataFrame()
        todo = []
        # select only the vectors in a given year.
        for k in range(len(to_process_dir_list)):
            todo = todo + [(x,source_list[k],to_process_dir_list[k]) for x in os.listdir(to_process_dir_list[k]) if (int(x.split('_')[0]) == year_todo)]

        for x in tqdm.tqdm(todo, 'merge the vectors'):
            f, k, to_process_dir = x
            vec = pd.read_pickle(to_process_dir + f).reset_index()
            vec['date'] = vec['timestamp'].apply(lambda x: x.split('T')[0])
            vec['date'] = pd.to_datetime(vec['date'], errors='coerce')
            if '_mean' in f:
                vec['vec_mean'] = 1
            else:
                vec['vec_mean'] = 0
            # now w reduce to the dates that have at least one eightk!
            vec = vec.merge(df)
            vec['reuters'] = (k =='ref')*1
            save_dest = save_dir + f.replace('.p', f'_{k}_data.tfrecord')
            print(save_dest)
            if vec.shape[0] > 0:
                with tf.io.TFRecordWriter(save_dest) as writer:
                    for _, row in vec.iterrows():
                        example = serialize_news_link_all_single_news(row)
                        writer.write(example)
    print('It all ran', flush=True)
