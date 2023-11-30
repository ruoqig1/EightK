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

if __name__ == '__main__':
    args = didi.parse()
    if Constant.IS_VM:
        start_dir = '/mnt/layline/logs/2017/'
        temp_dir = f'temp/log/unzip{args.a}/'
        final_dir_1 = 'res/log/high_freq/'
    else:
        start_dir = 'data/log/zip/'
        temp_dir = f'temp/log/unzip{args.a}/'
        final_dir_1 = 'res/log/high_freq/'
    os.makedirs(final_dir_1,exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    # load the ati filters to keep only proper match to crsp
    data = Data(Params())
    # unlike the previous 01_log_unzip, we will merge ati on date to keep only download on day the form is out
    ati_list = data.load_ati_cleaning_df().rename(columns={'adate':'date','form_id':'accession'})
    ati_list = ati_list[['date','accession','permno']]

    todo = np.array_split(np.sort(os.listdir(start_dir)),100)[args.a]
    for f in tqdm.tqdm(todo):
        unzip_file(start_dir+f,temp_dir)
        csv_file = [x for x in os.listdir(temp_dir) if '.csv' in x][0]
        df = pd.read_csv(temp_dir+csv_file)

        df['date'] = pd.to_datetime(df['date'])
        df['time']=df['time'].apply(lambda x: int(str(x).replace(':','')))
        df = df.merge(ati_list,left_on=['accession','date'],right_on=['accession','date'],how='inner')
        df = df.groupby(['ip','date','accession','permno','crawler'])['time'].min().reset_index()
        tot=df.groupby(['date','time','accession'])['ip'].nunique().reset_index()

        tot.to_pickle(final_dir_1+f.replace('.zip','.p'))
        print('saved file',f.replace('.zip','.p'))
        shutil.rmtree(temp_dir)








