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

if __name__ == '__main__':
    par = Params()
    args = didi.parse()
    data = Data(par)

    par.enc.opt_model_type = OptModelType.BOW1
    par.enc.news_source = NewsSource.NEWS_THIRD
    load_dir = par.get_cosine_dir(temp=False)
    df = pd.read_pickle(load_dir+'df.p')
    prn = pd.read_pickle('data/prn.p').rename(columns={'id':'news_id'})
    df =df.merge(prn,how='left')
    df['prn']=df['prn'].fillna(False)
    df = df.loc[df['prn']==False,:]

    df = df.loc[df['news_prov']==1,:]  # keep only reutersz


    firm_to_keep = df.loc[df['dist'] == 0, :].groupby('permno')['value'].mean()
    firm_to_keep = list(firm_to_keep[firm_to_keep.values > .05].index)
    ind = df['permno'].isin(firm_to_keep)
    print('Keep', ind.mean())
    df = df.loc[ind,:]

    # ## ad the items number to drop wrong ones
    # ev=data.load_list_by_date_time_permno_type()[['adate','permno','items']].rename(columns={'adate':'form_date'})
    # ev['form_date'] = pd.to_datetime(ev['form_date'])
    # df['permno'] = df['permno'].astype(int)
    # df = df.merge(ev)
    # ind = df['items'].isin(Constant.LIST_ITEMS_TO_USE)
    # print('Keep with items', ind.mean())
    # df = df.loc[ind,:]
    # # remove duplicated across items
    # df = df.drop(columns=['items']).drop_duplicates()
    #
    # normalise to keep the highest copy past news of the days by reuters
    df= df.loc[:, :].groupby(['news_date', 'permno','dist'])[['value','form_date']].max().reset_index()




    #
    # df['year']=df['form_date'].dt.year
    # ind = df['dist']==0
    # df.loc[ind,:].groupby(['year','permno'])['value'].count().reset_index().groupby('year')['value'].median().plot()
    # plt.show()

    indu = pd.read_csv(data.raw_dir+'industry.csv')
    indu.columns = ['permno','indu','indu2']
    indu = indu.dropna()
    indu['code']= indu['indu'].apply(lambda x: int(str(x)[:4]))
    indu['tech'] = indu['code'].isin([5132, 5182, 5192, 5415])


    mcap = data.load_mkt_cap_yearly()
    df['year'] = df['form_date'].dt.year
    df =df.merge(mcap)

    df.loc[df['dist']==0,'value'].hist(bins=100)
    plt.show()

    # df.loc[df['dist']==0,:].groupby(['news_date','permno'])['value'].max().hist(bins=100)
    # plt.show()


    df.groupby(['dist','permno'])['value'].mean().reset_index().groupby('dist')['value'].mean().plot()
    plt.show()

    df['big'] = df['mcap_d']==10
    mcap_col = 'big'
    t=df.groupby(['dist',mcap_col])['value'].mean().reset_index().pivot(columns=mcap_col,index='dist',values='value')
    t.plot()
    plt.show()

    df.loc[df['dist']==0].groupby(['mcap_d'])['value'].mean().plot()
    plt.show()

    # crsp = data.load_crsp_daily()
    # crsp['ret'] = pd.to_numeric(crsp['ret'],errors='coerce')
    # crsp = crsp.dropna().sort_values(['permno','date']).reset_index(drop=True)
    # crsp['mom'] = crsp.groupby('permno')['ret'].transform(lambda x: x.rolling(60).mean())
    # crsp['mom'] = crsp.groupby('permno')['mom'].transform(lambda x: x.shift(-3))
    # crsp['ev'] = crsp.groupby('permno')['ret'].transform(lambda x: x.rolling(3).mean())
    # crsp['ev'] = crsp.groupby('permno')['ev'].transform(lambda x: x.shift(-1))
    # crsp['mom_p'] = crsp.groupby('date')['mom'].rank(pct=True)
    # crsp =crsp.dropna()
    # crsp['mom_d'] = np.ceil(crsp['mom_p'] * 10).astype(int)
    # crsp = crsp.rename(columns={'date':'form_date'})
    # df = df.merge(crsp[['form_date','permno','ev','mom','mom_d']])
    #
    # df['ev_mag'] = df['ev'].abs().rank(pct=True)
    # df['ev_mag'] = np.ceil(df['ev_mag'] * 10).astype(int)
    #
    # df['ev_sign'] = np.sign(df['ev']).replace({0:1})
    #
    # df.loc[df['dist']==0].groupby(['ev_mag'])['value'].mean().plot()
    # plt.show()
    # df.loc[df['dist']==0].groupby(['ev_sign'])['value'].mean().plot()
    # plt.show()

    ev=data.load_list_by_date_time_permno_type()[['adate','permno','items']].rename(columns={'adate':'form_date'})
    ev['form_date'] = pd.to_datetime(ev['form_date'])
    df['permno'] = df['permno'].astype(int)


    ev = df.merge(ev)
    ev = ev.loc[ev['dist']==0,:]

    t=pd.DataFrame(ev.groupby('items')['value'].mean())
    t_count=pd.DataFrame(ev.groupby('items')['value'].count())
    t_count.columns =['count']
    t=t[t_count['count'].values>1000]
    t_count=t_count[t_count['count'].values>1000]

    big_items_by_items_plots(t)
    plt.show()

    big_items_by_items_plots(t_count)
    plt.show()

    ev['year'] = ev['form_date'].dt.year
    t=ev.groupby(['year','items'])['value'].count().reset_index().pivot(columns='items',values='value',index='year').fillna(0.0)

    col = t.sum()
    t[col[col>1000].index].plot()
    t = t/t.sum(1).values.reshape(-1,1)
    t.plot()
    plt.show()
    df = df.merge(indu)

    for big in [True,False]:
        t=df.loc[df['big']==big,:].groupby(['tech','dist'])['value'].mean().reset_index().pivot(columns='tech',index='dist',values='value')
        t.plot()
        plt.title(f'Big = {big}')
        plt.tight_layout()
        plt.show()


    # delist = pd.read_csv(data.raw_dir+'delisting.csv')
    # delist.columns = ['permno','date_bank','dl']
    # delist['date_bank'] = pd.to_datetime(delist['date_bank'])
    # ind = delist['dl'].isin([450.0,470.0,480.0])
    # delist = delist.loc[ind,:]
    # #
    # df = pd.read_pickle(load_dir+'df.p')
    #
    # delist = delist.merge(df)
    # delist['year'] = delist['form_date'].dt.year
    # delist = delist.merge(mcap)
    # delist['dist_b'] =( delist['date_bank']-delist['form_date']).dt.days
    #
    # ind = (delist['dist']==0) &(delist['mcap_d']>-1) &(delist['dist_b']<=365)
    # plt.scatter(delist.loc[ind,'dist_b'],delist.loc[ind,'value'])
    # plt.show()
    #
    # temp = delist.loc[ind,:].dropna()
    # temp['const'] = 1.0
    #
    # sm.OLS(temp[['value']],temp[['const','dist']]).fit().summary()


    df['year'] = df['form_date'].dt.year
    df['ym'] = df['form_date'].dt.year*100 + df['form_date'].dt.month
    df['ym'] = pd.to_datetime(df['ym'], format='%Y%m')

    temp = df.loc[df['dist']==0,:]
    # temp = temp.loc[temp['value']>=0.05,:]
    ind = temp['mcap_d']<=10
    temp.loc[ind,:].groupby('year')['value'].mean().plot()
    plt.show()



    temp['big'] = temp['mcap_d']>=10
    gb_col ='big'

    temp.groupby(['year',gb_col])['value'].quantile(.5).reset_index().pivot(columns=gb_col,index='year',values='value').plot()
    plt.show()
    temp.groupby(['year',gb_col])['value'].mean().reset_index().pivot(columns=gb_col,index='year',values='value').plot()
    plt.show()


    #
    # t=temp.groupby(['year', 'big','permno'])['value'].count().reset_index().groupby(['year', 'big'])['value'].mean().reset_index().pivot(columns='big', index='year', values='value')
    # t.plot()
    # plt.show()

    ### ADD ABNORMAL RETURNS FOR SHITS AND GIGGLE
    ev = pd.read_pickle(data.p_dir + 'abn_ev_monly.p')
    temp = temp.merge(ev.loc[ev['evttime']==0,['abs_abret','ret','abret','permno','date']])
    temp['abs_abret']=PandasPlus.winzorize_series(temp['abret'].abs(),1)


    temp = temp.dropna()
    plt.scatter(temp['value'],temp['abs_abret'],color='k',marker='+')
    plt.show()

    temp['r_cosine'] = np.round(10*temp['value'])/10
    temp.groupby(['date','r_cosine'])['abs_abret'].max().reset_index().groupby('r_cosine')['abs_abret'].mean().plot()
    plt.show()


