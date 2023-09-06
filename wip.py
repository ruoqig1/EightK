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
from clean_eightk import is_readable


if __name__ == "__main__":
    load_dir = 'res/8k_clean/'
    year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(load_dir)]))
    id_cols = ['cik','form_id']
    df = pd.DataFrame()
    for year in tqdm.tqdm(year_list,'Merge All the years'):
        press =pd.read_pickle(load_dir+f'press_{year}.p')[id_cols].drop_duplicates()
        legal =pd.read_pickle(load_dir+f'legal_{year}.p')[id_cols].drop_duplicates()
        press['release'] = True
        legal = legal.merge(press,how='left').fillna(False)
        legal['year']=year
        df = pd.concat([df,legal],axis=1)