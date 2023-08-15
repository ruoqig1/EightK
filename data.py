import os
import time
import datetime

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
            df = df.drop(columns=['acceptanceDatetime','filingDate'])

            tr = pd.read_csv(self.raw_dir +'cikPermno.csv')
            tr['namedt'] = pd.to_datetime(tr['namedt'])
            tr['enddat'] = pd.to_datetime(tr['enddat'])
            tr['begdat'] = pd.to_datetime(tr['begdat'])
            ind = tr['enddat']>=df['fdate'].min()
            tr = tr.loc[ind,:]

            tr= tr[['cik','permno','begdat','enddat']]

            # Merge and filter
            result = pd.merge(df[['fdate','cik']], tr, on='cik', how='left')
            result = result[(result['fdate'] >= result['begdat']) & (result['fdate'] <= result['enddat'])]
            result = result[['fdate','cik','permno']].dropna().drop_duplicates()

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

            df = df[['items', 'fdate','adate','atime', 'year', 'permno']].explode('items').reset_index(drop=True)
            df['cat'] = np.floor(df['items'])
            df.groupby('cat')['permno'].count().sort_values()

            df.to_pickle(self.p_dir+'load_list_by_date_time_permno_type.p')

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


            t = df[['permno', 'fdate']].dropna().drop_duplicates()
            t['permno'] = t['permno'].astype(int)
            t.to_csv(self.raw_dir+'eve_input.txt', sep=' ', index=False, header=False)

            t = df[['ticker']].dropna().drop_duplicates()
            t.to_csv(self.raw_dir+'tickers_list_for_ravenpack.txt', sep=' ', index=False, header=False)
        else:
            df = pd.read_pickle(self.p_dir+'load_list_by_date_time_permno_type.p')

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



if __name__ == "__main__":
    try:
        grid_id = int(sys.argv[1])
        print('Running with args',grid_id,flush=True)
    except:
        print('Debug mode on local machine')
        grid_id = -2

    self = Data(Params())
    reload = True

