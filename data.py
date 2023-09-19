import os
import time
import datetime

import pyperclip
import yfinance as yf

import numpy as np
import pandas as pd
import tqdm
import os
import glob
import gzip
from parameters import Params, Constant
import html2text
import sys
import re
from didipack import PandasPlus
import wrds
import didipack as didi



def filter_items(x):
    if (len(x)==1):
        x = [float(x)]
    elif '|' in x:
        x = [float(i) for i in x.split('|')]
    elif len(x)==2:
        x = [float(x)/10]
    else:
        x = [float(x)]
    return x


class Data:
    def __init__(self, par: Params):
        self.par = par

        self.raw_dir = par.data.base_data_dir + '/raw/'
        self.p_dir = par.data.base_data_dir + '/p/'
        self.p_eight_k_clean = par.data.base_data_dir + '/cleaned/eight_k_first_process/'
        self.p_eight_k_token = par.data.base_data_dir + '/cleaned/eight_k_tokens/'
        self.p_news_year = par.data.base_data_dir + '/cleaned/news_year/'
        self.p_news_tickers_related = par.data.base_data_dir + '/cleaned/news_tickers_related/'
        self.p_news_third_party_year = par.data.base_data_dir + '/cleaned/news_year_third_party/'
        self.p_news_token_year = par.data.base_data_dir + '/cleaned/news_token_year/'
        self.p_news_third_party_token_year = par.data.base_data_dir + '/cleaned/news_third_party_token_year/'
        self.p_vec_refinitiv = par.data.base_data_dir + '/vector/refinitiv/'
        self.p_vec_third_party = par.data.base_data_dir + '/vector/third_party/'

        os.makedirs(self.p_news_tickers_related,exist_ok=True)
        os.makedirs(self.p_vec_refinitiv,exist_ok=True)
        os.makedirs(self.p_vec_third_party,exist_ok=True)

        os.makedirs(self.p_news_third_party_token_year,exist_ok=True)
        os.makedirs(self.p_news_third_party_year,exist_ok=True)
        os.makedirs(self.p_news_token_year,exist_ok=True)
        os.makedirs(self.p_news_year,exist_ok=True)
        os.makedirs(self.p_eight_k_token,exist_ok=True)
        os.makedirs(self.p_eight_k_clean,exist_ok=True)
        os.makedirs(self.raw_dir,exist_ok=True)
        os.makedirs(self.p_dir,exist_ok=True)

        self.ROW_IN_FULL_RAVENPACK = 114351805

    def load_icf_current(self,reload = False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'lcf_current.csv')
            df['acceptanceDatetime'] = pd.to_datetime(df['acceptanceDatetime'].astype(str).str[:-2], format='%Y%m%d%H%M%S', errors='coerce')
            df['atime']=df['acceptanceDatetime'].dt.time
            df['adate']=df['acceptanceDatetime'].dt.date
            df['fdate'] = pd.to_datetime(df['filingDate'], format='%Y%m%d')
            df['first_date'] = pd.to_datetime(df['period'].fillna(0).astype(int).astype(str).replace('0', 'NaT'), format='%Y%m%d', errors='coerce')

            tr = pd.read_csv(self.raw_dir +'cikPermno.csv')
            tr['namedt'] = pd.to_datetime(tr['namedt'])
            tr['enddat'] = pd.to_datetime(tr['enddat'])
            tr['begdat'] = pd.to_datetime(tr['begdat'])
            ind = tr['enddat']>=df['fdate'].min()
            tr = tr.loc[ind,:]

            tr= tr[['cik','permno','gvkey','begdat','enddat']]

            # Merge and filter
            result = pd.merge(df[['fdate','cik']], tr, on='cik', how='left')
            result = result[(result['fdate'] >= result['begdat']) & (result['fdate'] <= result['enddat'])]
            result = result[['fdate','cik','permno','gvkey']].dropna().drop_duplicates()

            df = df.merge(result)

            df.to_pickle(self.p_dir+'load_icf_complete.p')
        else:
            df = pd.read_pickle(self.p_dir+'load_icf_complete.p')
        return df

    def load_list_by_date_time_permno_type(self,reload = False):
        if reload:
            df = self.load_icf_current()

            np.sort(df['type'].unique())

            # select only proper 8k
            ind = df['type'] == '8-K'
            df = df.loc[ind, :]

            # build year col
            df['year'] = df['fdate'].dt.year
            df = df.dropna(subset=['items'])
            df['items'] = df['items'].apply(filter_items)

            df = df[['items','cik','accessionNumber', 'fdate','adate','atime', 'year', 'permno','gvkey','first_date']].explode('items').reset_index(drop=True).rename(columns={'accessionNumber':'form_id'})
            df['cat'] = np.floor(df['items'])
            df.groupby('cat')['permno'].count().sort_values()
            df['form_id'] = df['form_id'].apply(lambda x: x.replace('-',''))

            ######## add ticker
            df['permno'] = df['permno'].astype(int)
            crsp = self.load_crsp_daily()[['date', 'permno', 'ticker']]
            crsp['ys'] = PandasPlus.get_yw(crsp['date'])
            df['ys'] = PandasPlus.get_yw(df['fdate'])

            # mere on perfect day and permno match
            df = df.merge(crsp[['date', 'permno', 'ticker']].rename(columns={'date': 'fdate'}), how='left')

            # select the one with no match
            m1 = df.loc[pd.isna(df['ticker']), :].copy().drop_duplicates().drop(columns='ticker')
            m1['ys'] = PandasPlus.get_yw(m1['fdate'])
            # merge again on the week
            m2 = m1.merge(crsp[['ys', 'permno', 'ticker']], how='left').dropna()
            # merge again the two types
            df = pd.concat([df.loc[~pd.isna(df['ticker']), :], m2], axis=0)

            # drop now the few that have mre than one ticker per event
            ind = df.groupby(['permno', 'fdate'])['ticker'].transform('nunique') == 1
            df = df.loc[ind, :]

            #drop poorly defined items
            ind = ~pd.isna(df['items'].map(Constant.ITEMS))
            df=df.loc[ind,:]

            df['gvkey'] = df['gvkey'].astype(int)

            # save
            df.to_pickle(self.p_dir+'load_list_by_date_time_permno_type.p')

            t = df[['permno', 'fdate']].dropna().drop_duplicates()
            t['permno'] = t['permno'].astype(int)
            t.to_csv(self.raw_dir+'eve_input.txt', sep=' ', index=False, header=False)

            t = df[['gvkey']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir+'gvkey.txt', sep=' ', index=False, header=False)
            t = df[['permno']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir+'permno.txt', sep=' ', index=False, header=False)

            t = df[['ticker']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir+'tickers_list_for_ravenpack.txt', sep=' ', index=False, header=False)
        else:
            df = pd.read_pickle(self.p_dir+'load_list_by_date_time_permno_type.p')

        df = df.loc[~df['items'].isin([3.1, 1.03]), :]
        return df

    def load_e_ff_long(self, reload=False,window=20):
        if reload:
            df =pd.read_csv(self.raw_dir+f'e{window}ff_long.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df['evtdate'] = pd.to_datetime(df['evtdate'])
            df = df.drop(columns=['model'])
            df.to_pickle(self.p_dir+f'e{window}ff_long.p')
        else:
            df = pd.read_pickle(self.p_dir+f'e{window}ff_long.p')
        return df


    def load_e_ffshort(self, reload=True,window=3):
        if reload:
            df = pd.read_csv(self.raw_dir+f'e{window}ff_short.csv')
            df = df[['permno','evtdate','cret','car']].rename(columns={'evtdate':'fdate'})
            df['fdate'] = pd.to_datetime(df['fdate'])
        return df

    def build_list_of_permno_for_crsp(self):
        df = self.load_list_by_date_time_permno_type()
        df = df[['permno']].drop_duplicates()
        df['permno'] = df['permno'].astype(int)
        df.to_csv(self.raw_dir + 'permno_for_crsp.txt', sep=' ', index=False, header=False)

    def load_crsp_daily(self,reload =False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'crsp_eightk.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df.to_pickle(self.p_dir + 'crsp_eightk.p')
        else:
            df = pd.read_pickle(self.p_dir + 'crsp_eightk.p')
        return df

    def load_list_of_tickers_in_news_and_crsp(self):
        r = pd.read_pickle(self.p_dir + 'ticker_in_news_and_crsp.p')
        return r

    def load_ravenpack_to_permno_tr(self,reload=False):
        if reload:
            # Connect to WRDS:
            # The first time you connect, you'll be asked for your WRDS username and password.
            db = wrds.Connection(wrds_username='ADIDISHEIM')

            # You can save the credentials to avoid entering them every time:
            # db.create_pgpass_file()

            # WRDS query
            query = """
            SELECT DISTINCT a.permno, b.rp_entity_id 
            FROM crsp.dse AS a, 
                 rpna.wrds_company_names AS b
            WHERE (a.ncusip <> '') AND a.ncusip = SUBSTR(b.isin, 3, 8);
            """

            # Fetch the results into a pandas DataFrame
            df = pd.read_sql(query, db.connection)
            df.to_pickle(self.p_dir + 'permon_ravenpack_id.p')
            # Close the connection
            db.close()
        else:
            df =pd.read_pickle(self.p_dir+'permon_ravenpack_id.p')
        return df

    def load_ravenpack_chunk(self,i=0, reload=False):
        i+=1
        if reload:
            tr = self.load_ravenpack_to_permno_tr(False)
            # keeping only permno present in icf
            t = self.load_list_by_date_time_permno_type()
            tr = tr.merge(t[['permno']].drop_duplicates()).drop_duplicates()

            chunk_size = self.ROW_IN_FULL_RAVENPACK // 10
            # Read the CSV in chunks
            chunks = pd.read_csv(self.raw_dir + 'raven_full.csv', chunksize=chunk_size)

            for i, chunk in enumerate(chunks, 1):
                print(f"Processing chunk {i}")
                # Do whatever processing you need on each chunk here
                chunk.columns = [x.lower() for x in chunk.columns]
                chunk = chunk.rename(columns={'rpa_date_utc': 'rdate', 'rpa_time_utc': 'rtime'})
                chunk['rdate'] = pd.to_datetime(chunk['rdate'])
                chunk = chunk.merge(tr)
                chunk = chunk[
                    ['rdate', 'rtime', 'permno', 'relevance', 'event_sentiment_score', 'topic', 'group', 'type',
                     'category', 'news_type']]
                chunk.to_pickle(self.p_dir + f'ravenpack{i}.p')

        df = pd.read_pickle(self.p_dir + f'ravenpack{i}.p')
        return df

    def load_snp_const(self,reload=False):
        if reload:
            conn = wrds.Connection(wrds_username='ADIDISHEIM')
            sp500 = conn.raw_sql("""
                                    select a.*, b.date, b.ret
                                    from crsp.msp500list as a,
                                    crsp.msf as b
                                    where a.permno=b.permno
                                    and b.date >= a.start and b.date<= a.ending
                                    and b.date>='01/01/2000'
                                    order by date;
                                    """, date_cols=['start', 'ending', 'date'])
            sp500['start'] = pd.to_datetime(sp500['start'])
            sp500['ending'] = pd.to_datetime(sp500['ending'])
            sp500['date'] = pd.to_datetime(sp500['date'])
            sp500['permno'] = sp500['permno'].astype(int)
            sp500 = sp500.drop(columns='ret')
            sp500.to_pickle(self.p_dir+'load_snp_const.p')
        else:
            sp500 = pd.read_pickle(self.p_dir+'load_snp_const.p')
        return sp500


    def load_mkt_cap_yearly(self, reload=False):
        if reload:
            # load market cap
            crsp = self.load_crsp_with_vol(reload=False)
            crsp['year'] = crsp['date'].dt.year
            crsp['mcap'] = (crsp['prc'] * crsp['shrout']).abs()
            crsp = crsp.groupby(['permno', 'year'])['mcap'].max().reset_index()
            crsp['mcap_p'] = crsp.groupby('year')['mcap'].rank(pct=True)
            crsp = crsp.dropna()
            crsp['mcap_d'] = np.ceil(crsp['mcap_p'] * 10).astype(int)
            mcap=crsp[['permno','year','mcap_d','mcap']].drop_duplicates()
            mcap.to_pickle(self.p_dir+'mcap.p')
        else:
            mcap  = pd.read_pickle(self.p_dir+'mcap.p')
        return mcap


    def load_some_relevance_icf(self,reload=False):
        if reload:
            # load market cap
            crsp = self.load_crsp_daily()
            crsp['year'] = crsp['date'].dt.year
            crsp['mcap'] = (crsp['prc'] * crsp['shrout']).abs()
            crsp = crsp.groupby(['permno', 'year'])['mcap'].max().reset_index()
            crsp['mcap_p'] = crsp.groupby('year')['mcap'].rank(pct=True)
            crsp = crsp.dropna()
            crsp['mcap_d'] = np.ceil(crsp['mcap_p'] * 10).astype(int)

            # do the matching
            rav = self.load_ravenpack_all()
            ev = self.load_list_by_date_time_permno_type()
            ev['adate'] = pd.to_datetime(ev['adate'])
            ev = ev.dropna(subset=['adate'])
            rav = rav.dropna(subset=['rdate'])

            rav = rav[['rdate', 'rtime', 'permno', 'relevance', 'event_sentiment_score']].rename(columns={'rdate': 'adate'})
            rav = rav.sort_values(['adate', 'permno', 'relevance']).reset_index(drop=True)
            rav = rav.drop_duplicates()

            df = ev.merge(rav, how='left')
            df['relevance'] = df['relevance'].fillna(0.0)
            df.columns
            rel_max = df.groupby(['items', 'adate', 'atime', 'year', 'permno', 'cat','cik','form_id'])['relevance'].max().reset_index()
            rel_max['no_rel'] = (rel_max['relevance'] == 0) * 1
            rel_max = rel_max.loc[rel_max['year'] >= 2000, :]

            rel_max = rel_max.merge(crsp.drop_duplicates())

            rel_max.to_pickle(self.p_dir + 'rel_max.p')
        else:
            rel_max = pd.read_pickle(self.p_dir + 'rel_max.p')
        rel_max['permno'] = rel_max['permno'].astype(int)
        return rel_max


    def load_ravenpack_all(self, reload=False):
        if reload:
            df = pd.DataFrame()
            for i in range(10):
                t = self.load_ravenpack_chunk(i)
                df = pd.concat([df,t],axis=0)
            df.to_pickle(self.p_dir+'raven_full.p')
        else:
            df = pd.read_pickle(self.p_dir+'raven_full.p')
        return df

    def load_analyst(self,reload=True):
        if reload:
            tr = pd.read_csv(self.raw_dir + 'ibes_tr.csv')
            df = pd.read_csv(self.raw_dir + 'ibes_main.csv')
            # Ensure the date columns are in datetime format
            df['ANNDATS'] = pd.to_datetime(df['ANNDATS'])
            tr['sdate'] = pd.to_datetime(tr['sdate'])
            tr['edate'] = pd.to_datetime(tr['edate'])

            # Merge on 'TICKER'
            merged_df = df.merge(tr, on='TICKER', how='left')

            # Filter rows where 'ANNDATS' is between 'sdate' and 'edate'
            result = merged_df[(merged_df['ANNDATS'] >= merged_df['sdate']) & (merged_df['ANNDATS'] <= merged_df['edate'])]

            # Keep relevant columns from df and the PERMNO column
            final_df = result[df.columns.tolist() + ['PERMNO']]

            final_df.columns = [x.lower() for x in final_df.columns]
            final_df = final_df.rename(columns={
                'ireccd': 'rec_value',
                'amaskcd': 'analyst_id',
                'anndats': 'rdate',
                'anntims': 'rtime',
            })
            final_df = final_df[['rec_value', 'analyst_id', 'rdate', 'rtime', 'permno']]
            final_df = final_df.dropna()

            final_df['permno'] = final_df['permno'].astype(int)
            final_df.to_pickle(self.p_dir + 'analyst.p')
        else:
            final_df = pd.read_pickle(self.p_dir + 'analyst.p')
        return final_df

    def load_main_no_rel_data(self,reload=False):
        if reload:
            per = self.load_some_relevance_icf(reload=False)

            ev = self.load_e_ff_long(window=40, reload=False).rename(columns={'evtdate': 'adate'})

            ev = ev.sort_values(['permno', 'evttime']).reset_index(drop=True).dropna()

            ev['abret'] = PandasPlus.winzorize_series(ev['abret'], 1)
            ev['cabret'] = ev.groupby('uid')['abret'].transform('cumsum')

            # merge all and drop weird items
            df = ev.merge(per)
            df['items_name'] = df['items'].map(Constant.ITEMS)
            df = df.dropna(subset='items_name')
            # add time
            df['hours'] = df['atime'].apply(lambda x: int(str(x)[:2]))

            # define large firm vs all
            df['large_cap'] = (df['mcap_d'] == 10).map({True: 'Large Cap', False: 'The Rest'})

            df['abret_abs'] = df['abret'].abs()
            df.to_pickle(self.p_dir+'load_main_no_rel_data.p')
        else:
            df= pd.read_pickle(self.p_dir+'load_main_no_rel_data.p')
        return df


    def load_yahoo(self, ticker='sp', freq='M', reload=False):
        if reload:
            # Define the ticker symbol
            if ticker == 'sp':
                ticker_symbol = '^GSPC'
            elif ticker == '10y':
                ticker_symbol = '^TNX'
            elif ticker == '30y':
                ticker_symbol = '^TYX'
            elif ticker == '5y':
                ticker_symbol = '^FVX'
            elif ticker == 'spbond':
                ticker_symbol = '^SP500BDT'
            elif ticker == 'vix':  # Added this condition for VIX
                ticker_symbol = '^VIX'

            # Set the start and end dates for the historical data
            start_date = '1983-01-01'
            end_date = '2023-06-22'

            # Download the historical data from Yahoo Finance
            df = yf.download(ticker_symbol, start=start_date, end=end_date)
            df['prc'] = df['Adj Close']
            df = df['prc']
            df.index = pd.to_datetime(df.index)

            if freq == 'M':
                df = df.resample('M').last()
            elif freq == 'D':
                pass
            else:
                raise ValueError("Invalid frequency. Choose either 'D' for daily or 'M' for monthly.")

            df.to_pickle(self.p_dir + f'{ticker}_{freq}_yahoo.p')
        else:
            df = pd.read_pickle(self.p_dir + f'{ticker}_{freq}_yahoo.p')

        df.index.name = 'date'
        df.name = f'{ticker}'
        if freq == 'M':
            df.index = df.index +pd.offsets.MonthEnd(0)
        return df

    def load_crsp_with_vol(self,reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'crsp_with_vol.csv')
            df.columns = [x.lower() for x in df.columns]
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'permno', 'vol', 'ret','prc','shrout']]
            df.to_pickle(self.p_dir + 'crsp_with_vol.p')
        else:
            df = pd.read_pickle(self.p_dir + 'crsp_with_vol.p')
        return df

    def load_abret_and_abvol(self,reload=False,T = 100,lag_for_ab_computation = 10):
        if reload:
            df = self.load_crsp_with_vol(reload=False)
            df = df.sort_values(['permno', 'date']).reset_index(drop=True)
            df['abs_ret'] = df['ret'].abs()
            for v in ['ret', 'vol','abs']:
                df['l'] = df.groupby('permno')[v].shift(lag_for_ab_computation)
                f = df.groupby('permno')[v].shift(-1).values
                m = df.groupby('permno')['l'].rolling(T).mean()
                s = df.groupby('permno')['l'].rolling(T).std()
                df[f'ab{v}-1'] = (df['l'].values - m.values) / s.values
                df[f'ab{v}0'] = (df[v].values - m.values) / s.values
                df[f'ab{v}1'] = (f - m.values) / s.values

            df = df[['permno', 'date', 'vol', 'ret'] + list(df.columns)[-6:]]
            df.to_pickle(self.p_dir + f'abvol_abret_T{T}_l{lag_for_ab_computation}')
        else:
            df = pd.read_pickle(self.p_dir + f'abvol_abret_T{T}_l{lag_for_ab_computation}')
        return df

    def load_ff5(self):
        ff = pd.read_csv(self.raw_dir + 'ff5.csv')
        ff['date'] = pd.to_datetime(ff['date'])
        return ff

    def load_ff5_m(self):
        ff = pd.read_csv(self.raw_dir + 'ff5_m.csv').rename(columns={'dateff':'date'})
        ff['date'] = pd.to_datetime(ff['date'])
        return ff

    def load_fundamentals(self,reload = False):
        if reload:
            df =pd.read_csv(self.raw_dir+'fundamentals.csv')
            df['date'] = pd.to_datetime(df['datadate'])
            df['roa'] = df['niq']/df['atq']
            df=df[['date','gvkey','piq','roa']]

            tr = pd.read_csv(self.raw_dir+'crsp_gvkey.csv')
            tr.columns = ['permno','gvkey','ltype','permco','start','end']
            tr['start']=pd.to_datetime(tr['start'],errors='coerce')
            tr['end']=pd.to_datetime(tr['end'],errors='coerce')
            tr['end']=tr['end'].fillna(tr['end'].max()+pd.DateOffset(years=1))

            df=df.merge(tr)
            ind = (df['date']>=df['start']) & ((df['date']<=df['end']))
            df = df.loc[ind,:]

            crsp = self.load_crsp_with_vol()
            #compute momentum, 12 -1
            crsp = crsp.sort_values(['permno','date']).reset_index(drop=True)
            lag = 20
            mom=crsp.set_index('date').groupby('permno')['ret'].rolling(250-lag).mean().reset_index()
            mom['ret']=mom.groupby('permno')['ret'].transform('shift',lag)
            mom = mom.rename(columns={'ret':'mom'})

            crsp['mcap'] = (crsp['prc']*crsp['shrout']*1000).abs()
            df=crsp[['date','permno','mcap']].merge(df[['date','permno','piq','roa']],how='outer')
            df = df.sort_values(['permno','date']).reset_index(drop=True)
            for c in df.columns[3:]:
                df[c]=df.groupby('permno')[c].fillna(method='ffill')

            df['pe'] = df['mcap']/(df['piq']*1e6)

            #add fundamentals of mom
            df=df.merge(mom,how='left')

            #save
            df[['date','permno','pe','roa','mom','mcap']].dropna().to_pickle(self.p_dir+'fundamentals.p')
        else:
            df = pd.read_pickle(self.p_dir+'fundamentals.p')
        return df

    def load_rav_avg_coverage(self,reload=False):
        if reload:
            rav = self.load_ravenpack_all()
            rav = rav.sort_values(['permno', 'rdate']).reset_index(drop=True)
            rav['date'] = rav['rdate']
            rav['permon'] = rav['permno'].astype(int)
            rav['news']= 1
            crsp = self.load_crsp_with_vol()
            crsp = crsp.merge(rav[['date','permno','news']],how='left')
            crsp['news'] = crsp['news'].fillna(0.0)
            crsp=crsp.groupby(['date','permno'])['news'].max().reset_index()
            crsp = crsp.sort_values(['permno','date']).reset_index(drop=True)
            lag = 20
            mom = crsp.set_index('date').groupby('permno')['news'].rolling(250 - lag).mean().reset_index()
            mom['news'] = mom.groupby('permno')['news'].transform('shift', lag)
            mom['news'] = mom['news'].fillna(0.0)
            mom.to_pickle(self.p_dir+'avg_coverage.p')
        else:
            mom = pd.read_pickle(self.p_dir+'avg_coverage.p')
        return mom

    def get_press_release_bool_per_event(self,reload = False):
        if reload:
            load_dir = self.p_eight_k_clean
            year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(load_dir)]))
            id_cols = ['cik', 'form_id']
            df = pd.DataFrame()
            for year in tqdm.tqdm(year_list, 'Merge All the years'):
                press = pd.read_pickle(load_dir + f'press_{year}.p')[id_cols].drop_duplicates()
                legal = pd.read_pickle(load_dir + f'legal_{year}.p')[id_cols].drop_duplicates()
                press['release'] = True
                legal = legal.merge(press, how='left').fillna(False)
                legal['year'] = year
                df = pd.concat([df, legal], axis=0)
            df.to_pickle(self.p_dir+'get_press_release_bool_per_event.p')
        else:
            df = pd.read_pickle(self.p_dir+'get_press_release_bool_per_event.p')

        df['cik']=df['cik'].astype(int)
        return df

    def load_all_item_list_in_refinitiv(self):

        '''
        pip install xlrd

        pip install openpyxl
        '''

        df_items = pd.DataFrame()
        col_name = ["qcode", "qode" ,"desc"]
        for sheet in ['News Topics', 'Attributions', 'Named Items', 'Product Codes']:
            t = pd.read_excel('./data/refinitiv_help/NewsCodes_20230417.xls', sheet_name=sheet)
            if t.shape[1]>3:
                t=pd.concat([t.iloc[:,:2],t.iloc[:,[4]]],axis=1)
            t.columns = col_name
            df_items = pd.concat([df_items, t], axis=0)


        return df_items

if __name__ == "__main__":
    try:
        grid_id = int(sys.argv[1])
        print('Running with args',grid_id,flush=True)
    except:
        print('Debug mode on local machine')
        grid_id = -2

    self = Data(Params())
    reload = True

    self
