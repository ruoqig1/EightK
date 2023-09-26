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
from utils_local.llm import EncodingModel
from utils_local.zip import decompress_gz_file, unzip_all
from utils_local.vec_functions import vectorise_in_batch
from utils_local.general import *



if __name__ == "__main__":
    args = didi.parse() # --legal=0/1 --eight=0/1 |  19 variations for legal eight
    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    par.enc.opt_model_type = OptModelType.OPT_13b
    par.enc.news_source = NewsSource.EIGHT_LEGAL
    par.train.tnews_only = True
    load_dir = par.get_training_dir() # the ouptut of merging the vectors


    print('Start loading Df', flush=True)
    df = pd.read_pickle(load_dir + 'main_df.p')
    print('Loaded Df', flush=True)
    df = set_ids_to_eight_k_df(df, par)

    # for norm in [Normalisation.ZSCORE,Normalisation.MINMAX, Normalisation.RANK]:
    for norm in [Normalisation.RANK]:
        par.train.norm = norm
        save_dir = par.get_training_norm_dir()  # where we will save the whole file, it's zscore dependant
        x = np.load(load_dir + 'x.npy')
        x = pd.DataFrame(x)
        x = normalize(x, par)
        for year in tqdm.tqdm(np.sort(np.unique(df['date'].dt.year)),norm.name):
            ind = df['date'].dt.year==year
            df.loc[ind,:].to_pickle(save_dir+f'df_{year}.p')
            x.loc[ind,:].to_pickle(save_dir+f'x_{year}.p')






