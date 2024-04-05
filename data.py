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

import data
from parameters import *
import html2text
import sys
import re
from didipack import PandasPlus
import wrds
import didipack as didi
import pytz


def load_some_enc(par: Params):
    load_dir_and_file = par.get_vec_process_dir(merged_bow=True)
    df = pd.read_pickle(load_dir_and_file)
    return df


def filter_items(x):
    if (len(x) == 1):
        x = [float(x)]
    elif '|' in x:
        x = [float(i) for i in x.split('|')]
    elif len(x) == 2:
        x = [float(x) / 10]
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
        self.p_to_vec_main_dir = par.data.base_data_dir + '/to_vec/'
        self.p_some_news_dir = par.data.base_data_dir + '/cleaned/some_news/'
        self.p_bow_merged_dir = Constant.MAIN_DIR + f'data/bow_merged/'
        self.cosine_final = Constant.MAIN_DIR + f'data/cosine_final/'

        os.makedirs(self.cosine_final, exist_ok=True)
        os.makedirs(self.p_bow_merged_dir, exist_ok=True)
        os.makedirs(self.p_some_news_dir, exist_ok=True)
        os.makedirs(self.p_to_vec_main_dir, exist_ok=True)
        os.makedirs(self.p_news_tickers_related, exist_ok=True)
        os.makedirs(self.p_vec_refinitiv, exist_ok=True)
        os.makedirs(self.p_vec_third_party, exist_ok=True)

        os.makedirs(self.p_news_third_party_token_year, exist_ok=True)
        os.makedirs(self.p_news_third_party_year, exist_ok=True)
        os.makedirs(self.p_news_token_year, exist_ok=True)
        os.makedirs(self.p_news_year, exist_ok=True)
        os.makedirs(self.p_eight_k_token, exist_ok=True)
        os.makedirs(self.p_eight_k_clean, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.p_dir, exist_ok=True)

        self.ROW_IN_FULL_RAVENPACK = 114351805

    def load_icf_current(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'lcf_current.csv')
            df['acceptanceDatetime'] = pd.to_datetime(df['acceptanceDatetime'].astype(str).str[:-2], format='%Y%m%d%H%M%S', errors='coerce')
            df['atime'] = df['acceptanceDatetime'].dt.time
            df['adate'] = df['acceptanceDatetime'].dt.date
            df['fdate'] = pd.to_datetime(df['filingDate'], format='%Y%m%d')
            df['first_date'] = pd.to_datetime(df['period'].fillna(0).astype(int).astype(str).replace('0', 'NaT'), format='%Y%m%d', errors='coerce')

            tr = pd.read_csv(self.raw_dir + 'cikPermno.csv')
            tr['namedt'] = pd.to_datetime(tr['namedt'])
            tr['enddat'] = pd.to_datetime(tr['enddat'])
            tr['begdat'] = pd.to_datetime(tr['begdat'])
            ind = tr['enddat'] >= df['fdate'].min()
            tr = tr.loc[ind, :]

            tr = tr[['cik', 'permno', 'gvkey', 'begdat', 'enddat']]

            # Merge and filter
            result = pd.merge(df[['fdate', 'cik']], tr, on='cik', how='left')
            result = result[(result['fdate'] >= result['begdat']) & (result['fdate'] <= result['enddat'])]
            result = result[['fdate', 'cik', 'permno', 'gvkey']].dropna().drop_duplicates()

            df = df.merge(result)

            df.to_pickle(self.p_dir + 'load_icf_complete.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_icf_complete.p')
        return df

    def load_list_by_date_time_permno_type(self, reload=False):
        if reload:
            df = self.load_icf_current()

            # select only proper 8k
            ind = df['type'] == '8-K'
            df = df.loc[ind, :]

            # build year col
            df['year'] = df['fdate'].dt.year
            df = df.dropna(subset=['items'])
            df['items'] = df['items'].apply(filter_items)

            df = df[['items', 'cik', 'accessionNumber', 'fdate', 'adate', 'atime', 'year', 'permno', 'gvkey', 'first_date']].explode('items').reset_index(drop=True).rename(columns={'accessionNumber': 'form_id'})
            df['cat'] = np.floor(df['items'])
            df.groupby('cat')['permno'].count().sort_values()
            df['form_id'] = df['form_id'].apply(lambda x: x.replace('-', ''))

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

            # drop poorly defined items
            ind = ~pd.isna(df['items'].map(Constant.ITEMS))
            df = df.loc[ind, :]

            df['gvkey'] = df['gvkey'].astype(int)

            # save
            df.to_pickle(self.p_dir + 'load_list_by_date_time_permno_type.p')

            t = df[['permno', 'fdate']].dropna().drop_duplicates()
            t['permno'] = t['permno'].astype(int)
            t.to_csv(self.raw_dir + 'eve_input.txt', sep=' ', index=False, header=False)

            t = df[['gvkey']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir + 'gvkey.txt', sep=' ', index=False, header=False)
            t = df[['permno']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir + 'permno.txt', sep=' ', index=False, header=False)

            t = df[['ticker']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir + 'tickers_list_for_ravenpack.txt', sep=' ', index=False, header=False)
        else:
            df = pd.read_pickle(self.p_dir + 'load_list_by_date_time_permno_type.p')

        df = df.loc[~df['items'].isin([3.1, 1.03]), :]
        return df

    def load_e_ff_long(self, reload=False, tp='ff', window=20):
        if reload:
            df = pd.read_csv(self.raw_dir + f'e{window}{tp}_long.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df['evtdate'] = pd.to_datetime(df['evtdate'])
            df = df.drop(columns=['model'])
            df.to_pickle(self.p_dir + f'e{window}{tp}_long.p')
        else:
            df = pd.read_pickle(self.p_dir + f'e{window}{tp}_long.p')
        return df

    def load_e_ffshort(self, reload=False, window=3):
        df = pd.read_csv(self.raw_dir + f'e{window}ff_short.csv')
        df = df[['permno', 'evtdate', 'cret', 'car']].rename(columns={'evtdate': 'fdate'})
        df['fdate'] = pd.to_datetime(df['fdate'])
        return df

    def build_list_of_permno_for_crsp(self):
        df = self.load_list_by_date_time_permno_type()
        df = df[['permno']].drop_duplicates()
        df['permno'] = df['permno'].astype(int)
        df.to_csv(self.raw_dir + 'permno_for_crsp.txt', sep=' ', index=False, header=False)

    def load_crsp_daily(self, reload=False):
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

    def load_ravenpack_to_permno_tr(self, reload=False):
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
            df = pd.read_pickle(self.p_dir + 'permon_ravenpack_id.p')
        return df

    def load_ravenpack_chunk(self, i=0, reload=False):
        i += 1
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

    def load_snp_const(self, reload=False):
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
            sp500.to_pickle(self.p_dir + 'load_snp_const.p')
        else:
            sp500 = pd.read_pickle(self.p_dir + 'load_snp_const.p')
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
            mcap = crsp[['permno', 'year', 'mcap_d', 'mcap']].drop_duplicates()
            mcap.to_pickle(self.p_dir + 'mcap.p')
        else:
            mcap = pd.read_pickle(self.p_dir + 'mcap.p')
        return mcap

    def load_some_relevance_icf(self, reload=False):
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
            rel_max = df.groupby(['items', 'adate', 'atime', 'rtime', 'year', 'permno', 'cat', 'cik', 'form_id'])['relevance'].max().reset_index()
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
                df = pd.concat([df, t], axis=0)

            # Combine rdate and rtime into a single datetime column
            df['datetime_utc'] = pd.to_datetime(df['rdate'].astype(str) + ' ' + df['rtime'])

            # Convert to Eastern Time Zone
            eastern = pytz.timezone('US/Eastern')
            df['datetime_etz'] = df['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(eastern)
            df['rdate'] = pd.to_datetime(df['datetime_etz'].dt.date)
            df['rtime'] = df['datetime_etz'].dt.time

            df = df.drop(columns=['datetime_etz', 'datetime_utc'])
            df.to_pickle(self.p_dir + 'raven_full.p')
        else:
            df = pd.read_pickle(self.p_dir + 'raven_full.p')
        return df

    def load_analyst(self, reload=False):
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

    def load_main_no_rel_data(self, reload=False):
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
            df.to_pickle(self.p_dir + 'load_main_no_rel_data.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_main_no_rel_data.p')
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
            df.index = df.index + pd.offsets.MonthEnd(0)
        return df

    def load_crsp_with_vol(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'crsp_with_vol.csv')
            df.columns = [x.lower() for x in df.columns]
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'permno', 'vol', 'ret', 'prc', 'shrout']]
            df.to_pickle(self.p_dir + 'crsp_with_vol.p')
        else:
            df = pd.read_pickle(self.p_dir + 'crsp_with_vol.p')
        return df

    def load_abret_and_abvol(self, reload=False, T=100, lag_for_ab_computation=10):
        if reload:
            df = self.load_crsp_with_vol(reload=False)
            df = df.sort_values(['permno', 'date']).reset_index(drop=True)
            df['abs_ret'] = df['ret'].abs()
            for v in ['ret', 'vol', 'abs']:
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
        ff = pd.read_csv(self.raw_dir + 'ff5_m.csv').rename(columns={'dateff': 'date'})
        ff['date'] = pd.to_datetime(ff['date'])
        return ff

    def load_fundamentals(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'fundamentals.csv')
            df['date'] = pd.to_datetime(df['datadate'])
            df['roa'] = df['niq'] / df['atq']
            df = df[['date', 'gvkey', 'piq', 'roa']]

            tr = pd.read_csv(self.raw_dir + 'crsp_gvkey.csv')
            tr.columns = ['permno', 'gvkey', 'ltype', 'permco', 'start', 'end']
            tr['start'] = pd.to_datetime(tr['start'], errors='coerce')
            tr['end'] = pd.to_datetime(tr['end'], errors='coerce')
            tr['end'] = tr['end'].fillna(tr['end'].max() + pd.DateOffset(years=1))

            df = df.merge(tr)
            ind = (df['date'] >= df['start']) & ((df['date'] <= df['end']))
            df = df.loc[ind, :]

            crsp = self.load_crsp_with_vol()
            # compute momentum, 12 -1
            crsp = crsp.sort_values(['permno', 'date']).reset_index(drop=True)
            lag = 20
            mom = crsp.set_index('date').groupby('permno')['ret'].rolling(250 - lag).mean().reset_index()
            mom['ret'] = mom.groupby('permno')['ret'].transform('shift', lag)
            mom = mom.rename(columns={'ret': 'mom'})

            crsp['mcap'] = (crsp['prc'] * crsp['shrout'] * 1000).abs()
            df = crsp[['date', 'permno', 'mcap']].merge(df[['date', 'permno', 'piq', 'roa']], how='outer')
            df = df.sort_values(['permno', 'date']).reset_index(drop=True)
            for c in df.columns[3:]:
                df[c] = df.groupby('permno')[c].fillna(method='ffill')

            df['pe'] = df['mcap'] / (df['piq'] * 1e6)

            # add fundamentals of mom
            df = df.merge(mom, how='left')

            # save
            df[['date', 'permno', 'pe', 'roa', 'mom', 'mcap']].dropna().to_pickle(self.p_dir + 'fundamentals.p')
        else:
            df = pd.read_pickle(self.p_dir + 'fundamentals.p')
        return df

    def load_rav_avg_coverage(self, reload=False):
        if reload:
            rav = self.load_ravenpack_all()
            rav = rav.sort_values(['permno', 'rdate']).reset_index(drop=True)
            rav['date'] = rav['rdate']
            rav['permon'] = rav['permno'].astype(int)
            rav['news'] = 1
            crsp = self.load_crsp_with_vol()
            crsp = crsp.merge(rav[['date', 'permno', 'news']], how='left')
            crsp['news'] = crsp['news'].fillna(0.0)
            crsp = crsp.groupby(['date', 'permno'])['news'].max().reset_index()
            crsp = crsp.sort_values(['permno', 'date']).reset_index(drop=True)
            lag = 20
            mom = crsp.set_index('date').groupby('permno')['news'].rolling(250 - lag).mean().reset_index()
            mom['news'] = mom.groupby('permno')['news'].transform('shift', lag)
            mom['news'] = mom['news'].fillna(0.0)
            mom.to_pickle(self.p_dir + 'avg_coverage.p')
        else:
            mom = pd.read_pickle(self.p_dir + 'avg_coverage.p')
        return mom

    def get_press_release_bool_per_event(self, reload=False):
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
            df.to_pickle(self.p_dir + 'get_press_release_bool_per_event.p')
        else:
            df = pd.read_pickle(self.p_dir + 'get_press_release_bool_per_event.p')

        df['cik'] = df['cik'].astype(int)
        return df

    def load_all_item_list_in_refinitiv(self):

        '''
        pip install xlrd

        pip install openpyxl
        '''

        df_items = pd.DataFrame()
        col_name = ["qcode", "qode", "desc"]
        for sheet in ['News Topics', 'Attributions', 'Named Items', 'Product Codes']:
            t = pd.read_excel('./data/refinitiv_help/NewsCodes_20230417.xls', sheet_name=sheet)
            if t.shape[1] > 3:
                t = pd.concat([t.iloc[:, :2], t.iloc[:, [4]]], axis=1)
            t.columns = col_name
            df_items = pd.concat([df_items, t], axis=0)

        return df_items

    def load_return_for_nlp_on_eightk_OLD(self, reload=False):
        if reload:
            crsp = self.load_crsp_daily()
            crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
            ev = self.load_list_by_date_time_permno_type()
            ev = ev.rename(columns={'adate': 'date', 'atime': 'eight_time'})
            ev['date'] = pd.to_datetime(ev['date'])

            crsp['ret_f'] = crsp.groupby('permno')['ret'].shift(-1)
            crsp['ret_l'] = crsp.groupby('permno')['ret'].shift(1)
            crsp['ret_m'] = (crsp['ret'] + crsp['ret_f'] + crsp['ret_l']) / 3
            big_l_col = []
            for l in [5, 20, 60, 250]:
                big_l_col.append(f'ret_{l}')
                crsp[f'ret_{l}'] = crsp.groupby('permno')['ret'].apply(lambda x: x.rolling(window=l).mean()).reset_index(level=0, drop=True)
                crsp[f'ret_{l}'] = crsp.groupby('permno')[f'ret_{l}'].shift(-(l + 1))

            crsp = crsp[['permno', 'date', 'ticker', 'ret', 'ret_l', 'ret_f', 'ret_m'] + big_l_col].dropna(subset=['ret', 'ret_l', 'ret_f', 'ret_m'])

            df = ev[['items', 'cik', 'form_id', 'date', 'ticker', 'eight_time']].merge(crsp)
            ev = None;
            crsp = None

            rav = self.load_ravenpack_all()
            # drop news and eightk before 16
            ind = rav['rtime'].apply(lambda x: int(x[:2]) <= 16)
            rav = rav.loc[ind, :]
            rav['news0'] = (rav['relevance'] >= 1) * 1

            rav = rav.groupby(['rdate', 'permno'])[['news0']].max().reset_index()
            rav = rav.rename(columns={'rdate': 'date'})
            rav['permno'] = rav['permno'].astype(int)

            df = df.merge(rav, how='left')
            df['news0'] = df['news0'].fillna(0.0)
            df.to_pickle(self.p_dir + 'load_return_for_nlp_on_eightk.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_return_for_nlp_on_eightk.p')
        return df

    def load_abn_return(self, model=1, with_alpha=False,reload=False):
        if model == 1:
            df = pd.read_pickle(self.p_dir + 'abn_ev_m.p')
        if model == 2:
            df = pd.read_pickle(self.p_dir + 'abn_ev_monly.p')
        elif model == 6:
            if with_alpha:
                df = pd.read_pickle(self.p_dir + 'abn_ev6_long.p')
            else:
                df = pd.read_pickle(self.p_dir + 'abn_ev6_monly.p')
        elif model == 7:
            df = pd.read_pickle(self.p_dir + 'abn_ev6_long_monly.p')
        elif model == 3:
            df = pd.read_pickle(self.p_dir + 'abn_ev3_monly.p')
        elif model == 66:
            # load the ati csv (and covnert it if reload)
            if reload:
                df = pd.read_csv(Constant.DRAFT_1_CSV_PATH+'b01-30-currReportsPanel.csv')
                df = df[['acceptanceDatetime','permno','date_ret','ret','exret','evttime','abret7','se_residual','beta']]
                df = df.rename(columns={
                    'abret7':'abret',
                    'se_residual':'sigma_ra'
                })
                df['date'] = pd.to_datetime(df['acceptanceDatetime']).dt.date
                df = df[['evttime','abret','ret','date','permno','sigma_ra','date_ret']].drop_duplicates()
                df.to_pickle(self.p_dir+'abn_ati_66_version.py')
            else:
                df = pd.read_pickle(self.p_dir+'abn_ati_66_version.py')

        elif model == -1:
            # use volume
            df = pd.read_pickle(self.p_dir + 'trun_ev_monly.p')
            df['abs_abret'] = df['abret']  # just doing this simplificaiton so the code run similarly for volume
        return df

    def load_return_for_nlp_on_eightk(self, reload=False):
        if reload:
            ret = self.load_abn_return()
            ret = ret.loc[ret['evttime'].between(-1, 1), :]
            ret = ret.groupby(['permno', 'date'])['ret', 'abret'].mean()
            ret = ret.reset_index()

            ev = self.load_list_by_date_time_permno_type()
            ev = ev.rename(columns={'adate': 'date', 'atime': 'eight_time'})
            ev['date'] = pd.to_datetime(ev['date'])

            df = ev[['items', 'cik', 'form_id', 'date', 'permno', 'eight_time']].merge(ret)

            rav = self.load_ravenpack_all()
            # drop news and eightk before 16
            ind = rav['rtime'].apply(lambda x: int(x[:2]) <= 16)
            rav = rav.loc[ind, :]
            rav['news0'] = (rav['relevance'] >= 1) * 1

            rav = rav.groupby(['rdate', 'permno'])[['news0']].max().reset_index()
            rav = rav.rename(columns={'rdate': 'date'})
            rav['permno'] = rav['permno'].astype(int)

            df = df.merge(rav, how='left')
            df['news0'] = df['news0'].fillna(0.0)
            df.to_pickle(self.p_dir + 'load_return_for_nlp_on_eightk_new.p')
            print('saved', self.p_dir + 'load_return_for_nlp_on_eightk_new.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_return_for_nlp_on_eightk_new.p')
        return df

    def load_crsp_all(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'crsp_full.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df['mcap'] = df['shrout'] * df['prc']
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df = df[['permno', 'date', 'ret', 'vol', 'mcap']]
            df.to_pickle(self.p_dir + 'load_crsp_all.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_crsp_all.p')
        return df

    def load_main_cosine(self, type=None):
        if type is None:
            save_dir = Constant.MAIN_DIR + 'data/cosine/opt_model_typeOptModelType.BOW1news_sourceNewsSource.NEWS_THIRDnb_chunks100save_chunks_size500chunk_to_run_id1/'
            df = pd.read_pickle(save_dir + 'df.p')
        else:
            # type ran wsj_one_per_stock.
            df = pd.read_pickle(self.cosine_final + f'{type}.p')
        return df

    def load_crsp_low_shares(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'crsp_low_shares.csv')
            df.columns = [x.lower() for x in df.columns]
            df['date'] = pd.to_datetime(df['date'])
            df['mcap'] = df['shrout'] * df['prc']
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
            df = df[['permno', 'date', 'ret', 'vol', 'mcap']]
            df.to_pickle(self.p_dir + 'load_crsp_low_shares.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_crsp_low_shares.p')
        return df

    def load_prn(self, reload=False):
        if reload:
            start_dir = data.p_news_tickers_related
            save_dir = data.p_to_vec_main_dir + '/single_stock_news_to_vec/'
            os.makedirs(save_dir, exist_ok=True)
            crsp = data.load_crsp_daily()
            print('loaded CRSP')
            list_valid = data.load_list_of_tickers_in_news_and_crsp()
            list_valid = list(list_valid.drop_duplicates().values)
            f = 'sample.p'
            # for f in ['third']:
            for f in ['third']:
                print('Start working on ', f)
                df = pd.read_pickle(start_dir + f + '.p')

                ind = df['audiences'].apply(lambda l: any([':PRN'.lower() in str(x).lower() for x in l]))
                ind_p = df['provider'].apply(lambda x: ':PRN'.lower() in x.lower())
                df['prn'] = ind | ind_p
                df[['id', 'prn']].to_pickle(self.p_dir + 'prn.p')
        else:
            df = pd.read_pickle(self.p_dir + 'prn.p')
        return df

    def load_complement_id_for_tfidf_records(self, reload=False):
        if reload:
            par = Params()
            par.enc.opt_model_type = OptModelType.BOW1
            par.enc.news_source = NewsSource.NEWS_THIRD
            load_dir = par.get_cosine_dir(temp=False)
            df = pd.read_pickle(load_dir + 'df.p')
            df['permno'].unique().shape
            prn = self.load_prn().rename(columns={'id': 'news_id'})
            df = df.merge(prn, how='left')
            df['prn'] = df['prn'].fillna(False)
            # drop the press release from this
            df = df.loc[df['prn'] == False, :]
            df = df.merge(df.groupby(['permno', 'news_prov'])['value'].median().reset_index().rename(columns={'value': 'm_cosine'}))
            df = df.rename(columns={'value': 'cosine'})
            # df['permno'] = df['permno'].astype(int)
            df['prn'] *= 1
            df = df.groupby('news_id')[['cosine', 'm_cosine', 'news_prov', 'prn']].max().reset_index()
            df.to_pickle(self.p_dir + 'load_complement_id_for_tfidf_records.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_complement_id_for_tfidf_records.p')[['news_id', 'cosine', 'm_cosine']]
        return df

    def load_rav_coverage_split_by(self, reload=False):
        if reload:
            rav = self.load_ravenpack_all()
            rav['pr_r'] = rav['news_type'] == 'PRESS-RELEASE'
            rav['pr_r'] = rav['pr_r'].map({True: 'press', False: 'article'})
            temp = rav.groupby(['rdate', 'permno', 'pr_r'])['relevance'].max().reset_index()
            temp = temp.pivot(columns='pr_r', index=['rdate', 'permno'], values='relevance').fillna(0.0).reset_index()
            temp['permno'] = temp['permno'].astype(int)
            temp = temp.rename(columns={'rdate': 'date'})
            temp.to_pickle(self.p_dir + 'load_rav_coverage.p')
        else:
            temp = pd.read_pickle(self.p_dir + 'load_rav_coverage.p')
        return temp

    def load_turnover(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'turnover.csv')
            df.columns = [x.lower() for x in df.columns]
            df['shrout'] *= 1000
            df['turnover'] = (df['vol']) / df['shrout']
            df['p_vol'] = df['prc'] * df['vol']
            df['p_shrout'] = df['prc'] * df['shrout']
            m = df.groupby(['date'])[['p_vol', 'p_shrout']].sum().reset_index()
            m['turnover_m'] = m['p_vol'] / m['p_shrout']

            df = df[['date', 'permno', 'turnover']].merge(m[['date', 'turnover_m']])
            df = df.dropna()
            df.dtypes
            df.to_pickle(self.p_dir + 'turnover.p')
        else:
            df = pd.read_pickle(self.p_dir + 'turnover.p')
        return df

    def load_wsj(self):
        df = pd.read_pickle('/data/gpfs/projects/punim2039/bllm/data/all_articles_df.p')
        return df

    def load_old_cosine(self, reload=False):
        if reload:
            final_dir = Constant.DROPBOX_COSINE_DATA + '/'
            df = pd.read_csv(final_dir + 'cosine_data.csv')
            df.to_pickle(self.p_dir + 'old_cosine.p')
        else:
            df = pd.read_pickle(self.p_dir + 'old_cosine.p')
        return df

    def load_bryan_data(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'usa.csv')
            df = df.dropna(subset='permno')
            df['permno'] = df['permno'].astype(int)
            ev = self.load_some_relevance_icf()
            ind = df['permno'].isin(ev['permno'].unique())
            df = df.loc[ind, :]
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            ind = df['date'].dt.year >= 2004
            df = df.loc[ind, :]
            df.to_pickle(self.p_dir + 'load_bryan_data.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_bryan_data.p')
        return df

    def load_wsj_one_per_tickers(self, reload=False):
        if reload:
            list_valid = self.load_list_of_tickers_in_news_and_crsp()
            list_valid = list(list_valid.drop_duplicates().values)

            df = self.load_wsj()

            ind = df['tickers'].apply(lambda x: '</djn-company>' in str(x))
            df.loc[ind, 'tickers'] = df.loc[ind, 'tickers'].apply(lambda x: x.split('</djn-company>')[0])

            def extract_tickers(text):
                tickers = re.findall(r'<c>(.*?)<\/c>', text)
                tickers_valdi = []
                for tick in tickers:
                    if '.' not in tick:
                        tickers_valdi.append(tick)
                    else:
                        if tick.split('.')[1] in ['N', 'O']:
                            tickers_valdi.append(tick)
                ticker_in_crsp = []
                for tick in tickers_valdi:
                    if tick in list_valid:
                        ticker_in_crsp.append(tick)
                return ticker_in_crsp

            df.loc[ind, 'tickers'] = df.loc[ind, 'tickers'].apply(extract_tickers)

            df = df.loc[ind, :]
            ind = df['tickers'].apply(len).between(1, 5)
            df = df.loc[ind, :]
            df = df[['headline', 'text', 'docdate', 'timestamp', 'tickers']].explode('tickers')
            df['docdate'] = pd.to_datetime(df['docdate'], format='%Y%m%d')
            df = df.rename(columns={'text': 'body', 'docdate': 'date', 'tickers': 'ticker'})
            df.reset_index(drop=True).to_pickle(self.p_dir + 'wsj_one_per_ticker.p')
        else:
            df = pd.read_pickle(self.p_dir + 'wsj_one_per_ticker.p')
        return df

    def load_ml_forecast_draft_1(self, model_index=2):
        load_dir = 'res/model_final_res/'
        os.listdir(load_dir)
        df = pd.read_pickle(load_dir + f'new_{model_index}.p')
        par = Params()
        par.load(load_dir, f'/par_{model_index}.p')
        return df, par

    def load_control_coverage(self, reload=False):
        if reload:
            rav = self.load_ravenpack_all()
            df = rav.copy()
            df['ym'] = PandasPlus.get_ym(df['rdate'])
            df['cov_sum'] = df['relevance'] > 0
            df = df.groupby(['ym', 'permno'])['cov_sum'].sum().reset_index()
            df['cov_pct'] = df.groupby(['ym'])['cov_sum'].rank(pct=True)
            df = df.sort_values(['permno', 'ym']).reset_index()
            df['cov_pct_l'] = df.groupby(['permno'])['cov_pct'].shift(1)
            df['cov_sum_l'] = df.groupby(['permno'])['cov_sum'].shift(1)
            df.to_pickle(self.p_dir + 'load_control_coverage.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_control_coverage.p')
        return df

    def load_ati_cleaning_df(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'currReportsPanel.csv')
            df['acceptanceDate'] = np.floor(df['acceptanceDatetime'] / 1000000).astype(int)
            df['atime'] = df['acceptanceDatetime'].apply(lambda x: str(x)[8:])
            df = df.loc[df['stable'] == 1, :]
            df = df[['acceptanceDate', 'accessionNumber', 'permno','atime', 'mktEquityEvent']]
            df = df.rename(columns={'acceptanceDate': 'adate', 'accessionNumber': 'form_id', 'mktEquityEvent': 'mcap_e', 'avgVol': 'avg_vol'})
            df['adate'] = pd.to_datetime(df['adate'], format='%Y%m%d')
            df['permno'] = df['permno'].astype(int)
            df = df.reset_index(drop=True)
            df.to_pickle(self.p_dir + 'load_ati_cleaning_df.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_ati_cleaning_df.p')
        return df

    def load_ati_cleaning_df_long(self, reload=False):
        if reload:
            df = pd.read_csv(self.raw_dir + 'currReportsPanel.csv')
            df['acceptanceDate'] = np.floor(df['acceptanceDatetime'] / 1000000).astype(int)
            df = df.loc[df['crspUnique'] == 1, :]
            df = df[['acceptanceDate', 'accessionNumber', 'permno', 'mktEquityEvent']]
            df = df.rename(columns={'acceptanceDate': 'adate', 'accessionNumber': 'form_id', 'mktEquityEvent': 'mcap_e', 'avgVol': 'avg_vol'})
            df['adate'] = pd.to_datetime(df['adate'], format='%Y%m%d')
            df['permno'] = df['permno'].astype(int)
            df = df.reset_index(drop=True)
            df.to_pickle(self.p_dir + 'load_ati_cleaning_df_long.p')
        else:
            df = pd.read_pickle(self.p_dir + 'load_ati_cleaning_df_long.p')
        return df

    def load_icf_ati_filter(self, reload=False, training=False):
        if reload:
            # build year col
            df = pd.read_csv(self.raw_dir + 'lcf_current.csv')
            df['acceptanceDatetime'] = pd.to_datetime(df['acceptanceDatetime'].astype(str).str[:-2], format='%Y%m%d%H%M%S', errors='coerce')
            df['atime'] = df['acceptanceDatetime'].dt.time
            df['adate'] = df['acceptanceDatetime'].dt.date
            df['adate'] = pd.to_datetime(df['adate'])
            # df['fdate'] = pd.to_datetime(df['filingDate'], format='%Y%m%d')
            df['form_id'] = df['accessionNumber']

            df = df.dropna(subset=['items'])
            df['items'] = df['items'].apply(filter_items)

            if training:
                ati = self.load_ati_cleaning_df_long()
            else:
                ati = self.load_ati_cleaning_df()
            ati['form_id'] = ati['form_id'].apply(lambda x: x.replace('-', ''))
            df['form_id'] = df['form_id'].apply(lambda x: x.replace('-', ''))
            df = df[['adate', 'form_id', 'items', 'atime']].explode('items').reset_index(drop=True)
            df = df.dropna()
            df['atime'] = df['atime'].apply(lambda x: int(str(x).replace(':', '')))
            df = df.merge(ati)

            rav = self.load_ravenpack_all()
            rav['news0'] = 1 * (rav['relevance'] > 0)
            rav['permno'] = rav['permno'].astype(int)

            rav = rav.rename(columns={'rdate': 'adate'})
            # rav = rav.groupby(['permno', 'adate'])['news0'].max().reset_index()
            rav['rtime'] = rav['rtime'].apply(lambda x: int(str(x).split('.')[0].replace(':', '')))
            rav_main = rav.loc[rav['news0'] == 1, :].groupby(['permno', 'adate'])['rtime'].max().reset_index()
            rav_main['news0'] = 1.0
            rav_futur = rav_main.copy().rename(columns={'rtime':'rtime_f','news0':'news0_f'})
            # substrsact a day so that when merge it's the news of the next day
            rav_futur['adate']=rav_futur['adate']-pd.DateOffset(days=1)

            df = df.merge(rav_main, how='left')
            df = df.merge(rav_futur, how='left')
            df['news0'] = df['news0'].fillna(0.0)
            df['news0_f'] = df['news0_f'].fillna(0.0)
            df['news_with_time'] = df['news0'].values
            ind = (df['news0'] == 1) & (df['rtime'] < df['atime'])
            df.loc[ind, 'news_with_time'] = 0


            # add the returns for trainings
            ret = self.load_abn_return(1)
            ret = ret.loc[ret['evttime'].between(-1, 1), :]
            ret = ret.groupby(['permno', 'date'])['ret', 'abret'].mean()
            ret = ret.reset_index()
            df = df.rename(columns={'adate': 'date'}).merge(ret)
            if training:
                df.to_pickle(self.p_dir + 'load_icf_ati_filter_training.p')
            else:
                df.to_pickle(self.p_dir + 'load_icf_ati_filter.p')
        else:
            if training:
                df = pd.read_pickle(self.p_dir + 'load_icf_ati_filter_training.p')
            else:
                df = pd.read_pickle(self.p_dir + 'load_icf_ati_filter.p')
        return df

    def load_logs_tot(self):
        df = pd.read_pickle(self.p_dir + 'log_tot_down.p')
        return df

    def load_logs_ip(self):
        df = pd.read_pickle(self.p_dir + 'log_ip.p')
        return df

    def load_logs_ev_study(self):
        df = pd.read_pickle(self.p_dir + 'log_ev_study.p')
        return df

    def load_logs_high(self):
        df = pd.read_pickle(self.p_dir + 'log_high.p')
        return df

    def load_logs_ip_small(self):
        df = pd.read_pickle(self.p_dir + 'log_ip_small.p')
        return df

    def load_news0_post_ati_change(self,reload = False):
        if reload:
            rav = self.load_ravenpack_all()

            rav = rav.loc[rav['relevance'] > 0, :]
            rav = rav.rename(columns={'rdate': 'date'})
            rav['afternoon_news'] = rav['rtime'].apply(lambda x: int(str(x)[:2]) >= 16) * 1
            rav['afternoon_news'].mean()

            # we find news that are published during the day and market hours. So news that could influence price of a news coming out during or before market hours today
            news_on_same_day = rav.loc[rav['afternoon_news'] == 0, :].groupby(['date', 'permno'])['rtime'].max().reset_index()
            news_on_same_day['news_on_same_day'] = 1
            news_on_same_day = news_on_same_day.rename(columns={'rtime': 'rtime_same_day'})

            # we find the news that happens the next day (by pushing date back) and during market hours. So news that could influence price of an 8k published yesterday afternoon
            news_next_day = rav.loc[rav['afternoon_news'] == 0, ['date', 'permno']].drop_duplicates().copy()
            news_next_day['date'] = news_next_day['date'] - pd.DateOffset(days=1)
            news_next_day['news_next_day'] = 1.0

            # finally, for news published after market hours we also caputre evening news.
            news_on_afternoon = rav.loc[rav['afternoon_news'] == 1, :].groupby(['date', 'permno'])['rtime'].max().reset_index()
            news_on_afternoon['news_on_afternoon'] = 1
            news_on_afternoon = news_on_afternoon.rename(columns={'rtime': 'rtime_afternoon'})

            ati = self.load_ati_cleaning_df()
            ati = ati.rename(columns={'adate': 'date'})
            ati['atime'] = pd.to_datetime(ati['atime'], format='%H%M%S').dt.time
            ati = ati.merge(news_next_day, how='left')
            ati = ati.merge(news_on_same_day, how='left')
            ati = ati.merge(news_on_afternoon, how='left')

            ati['news_on_afternoon'] = ati['news_on_afternoon'].fillna(0.0)
            ati['news_on_same_day'] = ati['news_on_same_day'].fillna(0.0)
            ati['news_next_day'] = ati['news_next_day'].fillna(0.0)

            ind_afternoon_atime = ati['atime'].apply(lambda x: x.hour) >= 16
            # check if a news came out in the afternoon after an afternoon 8k
            n1 = (ind_afternoon_atime == True) & (ati['news_on_afternoon'] == 1) & (ati['rtime_afternoon'] >= ati['atime'])
            # check if a news came out during the day after an 8k during or before market hours
            n2 = (ind_afternoon_atime == False) & (ati['news_on_same_day'] == 1) & (ati['rtime_same_day'] >= ati['atime'])
            # check if a news came out the day after an afternoon pub and during next day market hours (accounted for earlier in code)
            n3 = (ind_afternoon_atime == True) & (ati['news_next_day'] == 1)

            ati['news0'] = (n1 | n2 | n3)

            ati[['date', 'form_id', 'news0']].to_pickle(self.p_dir + 'news0_post_ati_change.p')
        else:
            ati = pd.read_pickle(self.p_dir+'news0_post_ati_change.p')
        return ati



if __name__ == "__main__":
    try:
        grid_id = int(sys.argv[1])
        print('Running with args', grid_id, flush=True)
    except:
        print('Debug mode on local machine')
        grid_id = -2

    self = Data(Params())
    # self.load_news0_post_ati_change(reload=True)