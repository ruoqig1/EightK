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
    high_dir = 'res/log/high_freq/'


    df = pd.DataFrame()
    for f in tqdm.tqdm(os.listdir(high_dir),'merging high frequ'):
        try:
            temp = pd.read_pickle(high_dir+f)
            df = pd.concat([temp,df],axis=0)
        except:
            print('failed with', f,flush=True)
    df.to_pickle(data.p_dir+'log_high.p')



