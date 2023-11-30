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
    par = Params()
    data = Data(par)
    ip_dir = 'res/log/ip/'
    tot_dir = 'res/log/tot/'



    df = pd.DataFrame()
    k = 0
    for f in tqdm.tqdm(os.listdir(ip_dir),'merging ip'):
        try:
            temp = pd.read_pickle(ip_dir+f)
            df = pd.concat([temp,df],axis=0)
            k+=1
        except:
            print('failed with', f,flush=True)
        if k == 100:
            df.to_pickle(data.p_dir+'log_ip_small.p')
            print('SAVED SMALL',flush=True)
    df.to_pickle(data.p_dir+'log_ip.p')



    df = pd.DataFrame()
    k = 0
    for f in tqdm.tqdm(os.listdir(tot_dir),'merging tot'):
        try:
            temp = pd.read_pickle(tot_dir+f)
            df = pd.concat([temp,df],axis=0)
            k+=1
        except:
            print('failed with', f,flush=True)

    df.to_pickle(data.p_dir+'log_tot_down.p')





