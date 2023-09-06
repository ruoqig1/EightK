import pandas as pd
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
import pyperclip
from cleaning import load_csv

def get_description(txt):
    if '<DESCRIPTION>' in txt:
        desc=file_text.split('<DESCRIPTION>')[1].split('<')[0]
        k=1
    else:
        desc = np.nan
        k=0
    return desc, k

def get_link(file_name):
    csv = load_csv(False)
    ind1=(csv['accessionNumber']==file_name)
    print(csv.loc[ind1,'URL'].iloc[0])


def get_items(form_id,item_check = '2.02'):
    csv = load_csv(False)
    ind1=(csv['accessionNumber'].apply(lambda x: str(x.replace('-','')))==form_id)
    return item_check in csv.loc[ind1,'items'].iloc[0]




if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/8k_clean/'
    os.makedirs(save_dir,exist_ok=True)
    load_dir = '/Users/adidisheim/Dropbox/AB-AD_Share/current2023/8-K/'

    icf = load_csv()
    icf = icf.dropna(subset=['sic','accessionNumber'])


    res = []
    for l in tqdm.tqdm(icf.index):
        cik = str(int(icf.loc[l,'cik']))
        form_id = str(icf.loc[l,'accessionNumber'].replace('-',''))
        f = load_dir+ f"{cik}/{form_id}/form.txt"
        some_earnings = ('2.02' in icf.loc[l,'items'])*1
        link =icf.loc[l,'URL']

        if os.path.exists(f):
            form = open(f, "r").read()
            k = 0
            for file_text in form.split('<FILENAME>'):
                desc, k_add = get_description(file_text)
                k += k_add
                res.append(pd.Series({'cik': cik, 'form': form_id, 'doc_nb': k, 'desc': desc, '2.02': some_earnings,'link':link}))

    df = pd.concat(res,axis=1).T
    df=df.dropna()





    # ind = (df['doc_nb'].isin([1,2,3]) )& (df['2.02']==0)
    # t=df.loc[ind,:]
    # df.loc[ind,'desc'].unique()
    #
    # get_link(file_name='000110465923088596')
    #
    # csv = load_csv()
    # csv.loc[csv['accessionNumber']=='000100291023000101']
    # get_items('000100291023000101')
    #
    # tt=t.loc[t['doc_nb']==1,:]
    #
    # ''':keyword
    # release, press,
    #
    # '''
#





