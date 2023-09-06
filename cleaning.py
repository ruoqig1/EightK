import pandas as pd

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

def load_csv(reload=False):
    if reload:
        csv = pd.read_csv('/Users/adidisheim/Dropbox/AB-AD_Share/current2023/currentReports.csv')
        csv.iloc[194,0]
        csv['acceptanceDatetime'] = pd.to_datetime(csv['acceptanceDatetime'].astype(str).str[:-2], format='%Y%m%d%H%M%S', errors='coerce')
        csv['atime'] = csv['acceptanceDatetime'].dt.time
        csv['adate'] = pd.to_datetime(csv['acceptanceDatetime'].dt.date)
        csv = csv.dropna(subset='adate')
        ind=(csv['adate'].dt.year == 2023) & (csv['adate'].dt.month == 8)
        csv=csv.loc[ind,:]

        csv['accessionNumber'] = csv['accessionNumber'].apply(lambda x: str(x.replace('-', '')))


        csv.to_pickle('temp.p')
    else:
        csv = pd.read_pickle('temp.p')
    return csv

if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40


    save_dir = Constant.DROP_RES_DIR + f'/8k_clean/'
    os.makedirs(save_dir,exist_ok=True)
    load_dir = '/Users/adidisheim/Dropbox/AB-AD_Share/current2023/8-K/'


    csv.iloc[99,0]
    #
    # csv['accessionNumber'] = csv['accessionNumber'].apply(lambda x: x.replace('-',''))
    # https://www.sec.gov/Archives/edgar/data/60512/
    #
    # 20230815
    #
    # ind=csv['accessionNumber']=='000000217823000076'
    # csv.loc[ind,:]

    # with documents cleanly sep
    start_file = load_dir+'2178/000000217823000076/'

    # https://www.sec.gov/Archives/edgar/data/314661/000089843094000146/0000898430-94-000146.txt
    start_file = load_dir+'314661/000089843094000146/'

    # https://www.sec.gov/Archives/edgar/data/1000694/000100069423000050/0001000694-23-000050-index-headers.html
    start_file = load_dir+'1000694/000100069423000050/'
    #https://www.sec.gov/Archives/edgar/data/1021561/000114036123037383
    start_file = load_dir+'1021561/000114036123037383/'


    form = open(start_file+"form.txt", "r").read()
    print(form.split('<TEXT>')[1])

    pyperclip.copy(form.split('<FILENAME>')[1])


    pyperclip.copy(form)

    first_document = form.split('<TEXT>')[1].split('</TEXT>')[0]



    # corresponding url https://www.sec.gov/Archives/edgar/data/2178/000000217823000076/0000002178-23-000076-index-headers.html
#
# for i in range(100):
#     print(csv.iloc[i,0])
#
# '''
# PRESS RELEAS
# Some documents have names like Document 2 - file: a2q23earningspressreleasef.htm
# https://www.sec.gov/Archives/edgar/data/1000694/000100069423000050/0001000694-23-000050-index-headers.html
#
# '''