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

    model = EncodingModel(par)

    res = df[id_col].copy()
    res['vec_last'] = np.nan

    res = res.set_index(id_col)
    df = df.set_index(id_col).sort_index()

    save_chunk_of_index = np.array_split(df.index, int(np.ceil(df.shape[0] / save_size)))
    for save_id in range(len(save_chunk_of_index)):
        save_dest = save_dir + f'{year}_{int(save_id + start_save_id)}.p'
        if os.path.exists(save_dest):
            print(save_dest, 'already processed', flush=True)
        else:
            print('#'*50)
            print('Start working on',save_dest, f'({len(save_chunk_of_index)})')
            print('#'*50,flush=True)
            # system to do in a few batch
            index_todo = np.array_split(save_chunk_of_index[save_id], int(np.ceil(len(save_chunk_of_index[save_id]) / batch_size)))
            for ind in tqdm.tqdm(index_todo, f'Tokenise {save_dest}'):
                txt_list_raw = list(df.loc[ind, 'txt'].values)
                txt_list = []
                for i in range(len(txt_list_raw)):
                    if txt_list_raw[i] in [None, '']:
                        txt_list.append(' ')
                    else:
                        txt_list.append(txt_list_raw[i].encode('utf-8', 'ignore').decode('utf-8'))
                last_token_hidden_stage, _ = model.get_hidden_states_para(texts=txt_list)
                res.loc[ind, 'vec_last'] = pd.Series(last_token_hidden_stage, index=ind)
            res.dropna().to_pickle(save_dest)
            res = res.loc[pd.isna(res.values)]

# python vec_main.py 26 --legal=0 --eight=0 --news=1 --ref=1 --bow=1

if __name__ == "__main__":
    pass

