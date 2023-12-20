import gc
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
import torch





def vectorise_in_batch(id_col:tuple, df:pd.DataFrame, save_size:int, batch_size:int, par:Params, year:int, start_save_id = 0):
    """
    Vectorize textual data in batches and save the results.

    This function takes a DataFrame containing text data, transforms the text into vectors using a given model,
    and saves these vectors in batches. The saving and processing are both done in chunks to accommodate large datasets.

    Parameters:
    id_col (tuple): Column(s) to be used as the unique identifier for the DataFrame.
    df (pd.DataFrame): The DataFrame containing text data to be vectorized.
    save_size (int): The number of rows per saved file.
    batch_size (int): The number of rows to be processed in each batch. (in the GPU)
    par (Params): A custom Params object that contains directory information.
    year (int): Year information, used for naming saved files.
    start_id (int): default 0, a value to say the batch number minimum to save in
    Returns:
    None. Saves vectorized data as pickle files in the directory specified by the Params object.
    """

    # get save dir
    save_dir = par.get_vec_process_dir()
    print('about to save in',save_dir,flush=True)
    # python3 vec_main.py 10 --legal=0 --eight=0 --news=1 --third=0 --third=1 --bow=0 --small=1


    res = df[id_col].copy()
    res['vec_last'] = np.nan
    # if par.enc.opt_model_type != OptModelType.BOW1:
    #     res['vec_mean'] = np.nan

    res = res.set_index(id_col)
    df = df.set_index(id_col).sort_index()

    res_mean = res.copy()
    model = EncodingModel(par)
    save_chunk_of_index = np.array_split(df.index, int(np.ceil(df.shape[0] / save_size)))
    for save_id in range(len(save_chunk_of_index)):
        save_dest = save_dir + f'{year}_{int(save_id + start_save_id)}.p'
        save_dest_mean = save_dir + f'{year}_{int(save_id + start_save_id)}_mean.p'
        if os.path.exists(save_dest):
            print(save_dest, 'already processed', flush=True)
        else:
            print('#'*50)
            print('Start working on',save_dest, f'({len(save_chunk_of_index)})')
            print('#'*50,flush=True)
            # system to do in a few batch
            index_todo = np.array_split(save_chunk_of_index[save_id], int(np.ceil(len(save_chunk_of_index[save_id]) / batch_size)))
            last_mat = []
            mean_mat = []

            for ind in tqdm.tqdm(index_todo, f'Tokenise {save_dest}'):
                if batch_size >1:
                    txt_list_raw = list(df.loc[ind, 'txt'].values)
                    txt_list = []
                    for i in range(len(txt_list_raw)):
                        if txt_list_raw[i] in [None, '']:
                            txt_list.append(' ')
                        else:
                            txt_list.append(txt_list_raw[i].encode('utf-8', 'ignore').decode('utf-8'))
                    last_token_hidden_stage, mean_hidden_stage = model.get_hidden_states_para(texts=txt_list)
                    # res.loc[ind, 'vec_last'] = pd.Series(last_token_hidden_stage, index=ind)
                    res.loc[ind, 'vec_last'] = pd.Series(last_token_hidden_stage, index=ind)
                    if par.enc.opt_model_type != OptModelType.BOW1:
                        res_mean.loc[ind, 'vec_last'] = pd.Series(mean_hidden_stage, index=ind)
                        # res.loc[ind, 'vec_mean'] = pd.Series(mean_hidden_stage, index=ind)
                else:
                    txt = df.loc[ind, 'txt'].values[0]
                    txt =txt.encode('utf-8', 'ignore').decode('utf-8')
                    last_token_hidden_stage, mean_hidden_stage = model.get_hidden_states(txt)
                    # last_mat.append(last_token_hidden_stage)
                    # mean_mat.append(mean_hidden_stage)
                    # res.loc[ind, 'vec_last'] = 'done'
                    # START PROBLEM
                    # breakpoint()
                    res.loc[ind, 'vec_last'] = pd.Series([last_token_hidden_stage], index=ind)
                    res_mean.loc[ind, 'vec_last'] = pd.Series([mean_hidden_stage], index=ind)
                    # if par.enc.opt_model_type != OptModelType.BOW1:
                    #     res.loc[ind, 'vec_mean'] = pd.Series([mean_hidden_stage], index=ind)#
                # if par.enc.framework == Framework.PYTORCH:
                #     torch.cuda.empty_cache()
            res.dropna().to_pickle(save_dest)
            res = res.loc[pd.isna(res.values)]
            res_mean.dropna().to_pickle(save_dest_mean)
            res_mean = res_mean.loc[pd.isna(res_mean.values)]
            # END PROBLEM
            # res.dropna()['vec_last'].iloc[0]
            # res.dropna()['vec_mean'].iloc[0]

if __name__ == "__main__":
    pass

