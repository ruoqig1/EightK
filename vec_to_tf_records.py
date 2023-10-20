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

def serialize_example(row):
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
        'abret': tf.train.Feature(float_list=tf.train.FloatList(value=[abret_val]))
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
        ev = data.load_list_by_date_time_permno_type()
        ev['date'] = pd.to_datetime(ev['adate'])
        df = data.load_return_for_nlp_on_eightk()
        df = df[['permno', 'date', 'news0', 'ret', 'abret']].drop_duplicates()
        vec = pd.DataFrame()
        todo = [x for x in os.listdir(to_process_dir) if (int(x.split('_')[0]) == year_todo)]
        for f in tqdm.tqdm(todo, 'merge the vectors'):
            t = pd.read_pickle(to_process_dir + f).reset_index()
            t['date'] = t['timestamp'].apply(lambda x: x.split('T')[0])
            t['date'] = pd.to_datetime(t['date'], errors='coerce')
            # now w reduce to the dates that have at least one eightk!
            t = t.merge(ev[['date', 'ticker', 'permno']].drop_duplicates())

            save_dest = save_dir + f.replace('.p', '_data.tfrecord')
            if t.shape[0] > 0:
                vec = t.copy()
                vec = vec.merge(df)
                with tf.io.TFRecordWriter(save_dest) as writer:
                    for _, row in vec.iterrows():
                        example = serialize_example(row)
                        writer.write(example)
    print('It all ran', flush=True)
