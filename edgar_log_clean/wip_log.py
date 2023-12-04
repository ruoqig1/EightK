import pandas as pd
import tqdm
import sys
from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, PlotPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyperclip
from bs4 import BeautifulSoup
import re
from utils_local.zip import unzip_file
import shutil

def load_one(f):
    os.makedirs(temp_dir,exist_ok=True)
    unzip_file(start_dir + f, temp_dir)
    csv_file = [x for x in os.listdir(temp_dir) if '.csv' in x][0]
    df = pd.read_csv(temp_dir + csv_file,nrows=10000)
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['time'].apply(lambda x: int(str(x).replace(':', '')))
    shutil.rmtree(temp_dir)
    return df

if __name__ == '__main__':
    args = didi.parse()

    temp_dir = f'temp/log/unzip{args.a}/'
    start_dir = '/mnt/layline/logs/2017/'

    os.makedirs(temp_dir, exist_ok=True)
    # load the ati filters to keep only proper match to crsp
    data = Data(Params())
    ati_list = data.load_ati_cleaning_df().rename(columns={'adate':'form_date','form_id':'accession'})
    ati_list = ati_list[['form_date','accession','permno']]

    todo = np.sort(os.listdir(start_dir))
    # for k in [1000,2000,3000,5000,-1]:
    res = []
    for k in tqdm.tqdm(range(len(todo))):
        df = load_one(todo[k])
        res.append([df['date'].min(),df['zone'].unique()])
        # print(df['date'].min(),df['zone'].unique())



    res = pd.DataFrame(res).dropna()
    res.to_pickle(data.p_dir+'log_time_zone.p')










