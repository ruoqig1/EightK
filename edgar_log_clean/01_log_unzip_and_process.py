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
        final_dir_1 = 'res/log/ip/'
        final_dir_2 = 'res/log/tot/'
    else:
        start_dir = 'data/log/zip/'
        temp_dir = f'temp/log/unzip{args.a}/'
        final_dir_1 = 'res/log/ip/'
        final_dir_2 = 'res/log/tot/'
    os.makedirs(final_dir_1,exist_ok=True)
    os.makedirs(final_dir_2,exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    # load the ati filters to keep only proper match to crsp
    data = Data(Params())
    ati_list = data.load_ati_cleaning_df().rename(columns={'adate':'form_date','form_id':'accession'})
    ati_list = ati_list[['form_date','accession','permno']]

    todo = np.array_split(np.sort(os.listdir(start_dir)),100)[args.a]
    for f in tqdm.tqdm(todo):
        unzip_file(start_dir+f,temp_dir)
        csv_file = [x for x in os.listdir(temp_dir) if '.csv' in x][0]
        df = pd.read_csv(temp_dir+csv_file)

        df['date'] = pd.to_datetime(df['date'])
        df['time']=df['time'].apply(lambda x: int(str(x).replace(':','')))
        df = df.merge(ati_list,left_on='accession',right_on='accession',how='inner')
        df = df.groupby(['ip','date','accession','permno','crawler'])['time'].min().reset_index()
        tot=df.groupby(['date','accession'])['ip'].nunique().reset_index()

        df.to_pickle(final_dir_1+f.replace('.zip','.p'))
        tot.to_pickle(final_dir_2+f.replace('.zip','.p'))
        print('saved file',f.replace('.zip','.p'))
        shutil.rmtree(temp_dir)








