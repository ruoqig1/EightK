import smtplib

import numpy as np
import pandas as pd
import os
from parameters import *
from matplotlib import pyplot as plt
from data import *
from utils_local.nlp_ticker import *
from statsmodels import api as sm




if __name__ == "__main__":
    par = Params()
    data = Data(par)

    load_dir = data.p_to_vec_main_dir + '/single_stock_news_to_vec/'
    # load_dir = data.p_news_year
    ev = data.load_list_by_date_time_permno_type()
    ev['date'] = pd.to_datetime(ev['adate'])
    ev= ev[['date','ticker','atime']].dropna()

    res = pd.DataFrame()
    for f in tqdm.tqdm(os.listdir(load_dir)):
        df =pd.read_pickle(load_dir+f)
        df = df[['id','alert','timestamp','date','ticker']].merge(ev)
        res = pd.concat([res,df],axis=0)
        res.to_pickle(data.p_dir+'news_per_stock_id.p')