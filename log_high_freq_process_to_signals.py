import didipack as didi
import pandas as pd
from matplotlib import pyplot as plt
from data import Data
from parameters import *
import seaborn as sns
from utils_local.plot import plot_ev
from utils_local.general import table_to_latex_complying_with_attila_totally_unreasonable_demands
import pandas as pd
import pytz
import pytz
import pandas as pd
import datetime
from log_analysis_high_frequ import convert_time



if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    data = Data(par)
    save_dir = Constant.EMB_PAPER
    df = data.load_logs_high()
    df = df.rename(columns={'accession':'form_id'})
    df['form_id'] = df['form_id'].apply(lambda x: str(x).replace('-',''))
    ati = data.load_icf_ati_filter()

    df =df.merge(ati)

    # add the thing to converting time to remove the utc bug change
    bug = pd.read_pickle(data.p_dir+'log_time_zone.p')
    bug['zone'] = bug[1].apply(lambda x: x[0])
    bug['date'] = pd.to_datetime(bug[0])
    bug =bug.drop(columns=[0,1])
    df = df.merge(bug)
    df['zone']/=100
    df['zone'] = df['zone'].astype(int)



    df['time'] = df['time'].apply(convert_time)
    df['rtime'] = df['rtime'].apply(convert_time)
    # df['atime'] = df.groupby(['form_id', 'permno'])['time'].transform('min')

    df['time'] = pd.to_timedelta(df['time'],errors='coerce')
    df['rtime'] = pd.to_timedelta(df['rtime'],errors='coerce')
    df['atime'] = pd.to_timedelta(df['atime'],errors='coerce')
    df['time'] = df['time'] - pd.to_timedelta(df['zone'], unit='h')
    df['year'] = df['date'].dt.year



    # total number of covered arround covered date
    df['dist_to_coverage'] = ((df['time'] - df['rtime']).dt.total_seconds() / 3600).round()
    df['dist_to_form'] = ((df['time'] - df['atime']).dt.total_seconds() / 3600).round()
    df['dist_news_to_a'] = ((df['rtime'] - df['atime']).dt.total_seconds() / 3600).round()

    df['rhours']=df['rtime'].dt.components['hours']
    df['ahours']=df['atime'].dt.components['hours']
    df['hours']=df['time'].dt.components['hours']

    df['hours']