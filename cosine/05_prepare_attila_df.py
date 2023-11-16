from data import Data
from utils_local.general import *
from matplotlib import pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from didipack import PandasPlus
import didipack as didi
import tqdm
from utils_local.plot import big_items_by_items_plots
import pandas as pd
from statsmodels import api as sm

def get_news_time(x):
    try:
        t = x.split('-')[1]
        res = str(t)[0:6]
        # res = str(t)[:2]
    except:
        res = np.nan
    return res

if __name__ == '__main__':
    par = Params()
    args = didi.parse()
    data = Data(par)
    final_dir = Constant.DROPBOX_COSINE_DATA+'/'
    # old_df = pd.read_csv(final_dir+'cosine_data.csv')

    os.makedirs(final_dir,exist_ok=True)
    par.enc.opt_model_type = OptModelType.BOW1
    par.enc.news_source = NewsSource.NEWS_THIRD
    load_dir = par.get_cosine_dir(temp=False)
    df = pd.read_pickle(load_dir+'df.p')
    df['permno'].unique().shape
    prn =data.load_prn().rename(columns={'id':'news_id'})
    df =df.merge(prn,how='left')
    df['prn']=df['prn'].fillna(False)

    df['permno'] = df['permno'].astype(int)

    ev_id = data.load_list_by_date_time_permno_type()
    ev_id = ev_id[['permno','adate','atime']].rename(columns={'adate':'form_date','atime':'form_time'}).drop_duplicates()
    ev_id['form_date'] = pd.to_datetime(ev_id['form_date'])
    df = df.merge(ev_id,how='left')

    # df = df.loc[df['news_prov']==1,:]  # keep only reutersz

    # keep only data with some degfree of mathcing (measured by non zero cosine at some poitn)
    firm_to_keep = df.loc[df['dist'] == 0, :].groupby('permno')['value'].mean()
    firm_to_keep = list(firm_to_keep[firm_to_keep.values > .05].index)
    ind = df['permno'].isin(firm_to_keep)
    print('Keep', ind.mean())
    df = df.loc[ind,:]

    #
    # # keep only data with some degfree of mathcing (measured by non zero cosine at some poitn)
    # firm_to_keep = df.loc[df['dist'] == 0, :].groupby('permno')['value'].max()
    # firm_to_keep = list(firm_to_keep[firm_to_keep.values > .1].index)
    # ind = df['permno'].isin(firm_to_keep)
    # print('Keep', ind.mean())
    # df = df.loc[ind,:]

    mcap = data.load_mkt_cap_yearly()
    df['year'] = df['form_date'].dt.year
    df =df.merge(mcap,how='left')

    # add news time
    df['news_time']=df['news_id'].apply(get_news_time)
    df['news_time'] = pd.to_datetime(df['news_time'], format='%H%M%S').dt.time

    # add abnormal returns
    ev = pd.read_pickle(data.p_dir + 'abn_ev_monly.p')
    ev = ev.pivot(columns='evttime',index=['date','permno'],values='abret')
    ev.columns = [f't{x}' if x < 0 else f't+{x}' for x in ev.columns]
    ev = ev.reset_index()
    ev = ev.rename(columns={'date':'form_date'})
    df['permno']
    df = df.merge(ev,how='left')

    print(df.columns)


    # save for attila in csv format
    # df.to_csv(final_dir+'cosine_data.csv')

    text = """
    Columns:
        - form_date: acceptance date of the 8k fillings
        - form_time: acceptance time of the 8k fillings
        - news_date: date of the news (or prn) 
        - news_time: publication time of the news
        - t-x: these columns, from t-20 to t+20 gives the abnormal returns arround the form_date (20 days before and after)
        - dist: distance in number of days between form_date and form
        - value: cosine similartiy between the press release attached with the 8k filling and the news
        - prn: a flag equal to true if the news is published on Press Release News Wire (so likely a press release by the firm)
        - mcap_d: the market cap decile (computed year per year) to which the firm belongs
        - mcap: the average yearly marcet cap of each firm (used to build mcap_d)
        - news_prov: an id = to 1 if the news if from reuters, 2 if it's form a third party
    """
    with open(final_dir+'documentations.txt', 'w') as file:
        file.write(text.strip())
