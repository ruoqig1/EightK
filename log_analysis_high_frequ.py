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



def convert_rtime_to_etz(df):
    # Define the UTC and ETZ time zones
    utc_zone = pytz.utc
    etz_zone = pytz.timezone('US/Eastern')

    # Convert 'date' column to datetime (if not already) and adjust for timezone
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(utc_zone)

    # Combine 'date' and 'rtime', convert to ETZ, and extract the time part
    df['rtime'] = (df['date'] + df['rtime']).dt.tz_convert(etz_zone).dt.time

    return df






# Function to convert to HH:MM:SS format
def convert_time(t):
    t_str = str(t).zfill(6)  # Convert to string and ensure 6 characters
    if pd.isna(t):
        return f"00:00:00"
    else:
        return f"{t_str[0:2]}:{t_str[2:4]}:{t_str[4:6]}"


def plot_hours(df):


    ind = (df['dist_news_to_a']>=0) & (df['news0']>0) & (df['dist_news_to_a']<=24) & (df['rhours']>5)
    ind_uncovered = (df['news0']==0)
    temp = pd.DataFrame(df.loc[ind,:].groupby('rhours')['ip'].count()).rename(columns={'ip':'% Coverage'})
    temp['% of 8k Forms (covered)'] = df.loc[ind,:].groupby('hours')['ip'].count()
    temp['% of 8k Forms (un-covered)'] = df.loc[ind_uncovered,:].groupby('hours')['ip'].count()
    # temp['% of logs'] = df.loc[ind_uncovered,:].groupby('hours')['ip'].sum()
    temp = temp/temp.sum()
    temp.plot(figsize=(6.4 * 2, 4.8))
    plt.xticks(range(int(6), int(df['rhours'].max()) + 1))
    plt.xlabel('Hours')
    plt.ylabel('Percentage of Events')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir+'logs_high_hours_of_day_ss.png')
    plt.show()



import pandas as pd
import datetime

def daylight_saving_ranges(year):
    # Function to find the nth weekday in a given month and year
    def find_nth_weekday(year, month, weekday, n):
        date = datetime.date(year, month, 1)
        # How many days to add to get to the first occurrence of the weekday
        days_to_add = (weekday - date.weekday() + 7) % 7
        date += datetime.timedelta(days=days_to_add)
        # Adding the remaining weeks (n - 1, as we already are at the 1st occurrence)
        date += datetime.timedelta(weeks=n-1)
        return date

    # Find the second Sunday in March
    dst_start = find_nth_weekday(year, 3, 6, 2)  # 6 = Sunday

    # Find the first Sunday in November
    dst_end = find_nth_weekday(year, 11, 6, 1)  # 6 = Sunday

    # The day after DST ends is the start of standard time
    std_start = dst_end + datetime.timedelta(days=1)

    # The day before DST begins is the end of standard time
    std_end = dst_start - datetime.timedelta(days=1)

    return {'saving': (dst_start, dst_end),
            'standard': (std_start, std_end)}



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
    plot_hours(df)
    df[['date','rtime']].dtypes

    ## BIG PLOTS ON NUMBER OF LOGS WHEN COVERAGE ARRIVE (removing the bump of new info
    ind = df['dist_to_coverage'].between(-6,6)
    temp = df.loc[ind,:]
    temp = temp.loc[df['news0']==1,:].copy()
    temp_big = temp.copy()
    ind = (temp['dist_to_form']>0)& (temp['dist_news_to_a']>=0)
    # ind = (temp['dist_to_form']>0) & (temp['dist_news_to_a']>0)
    temp_big = temp.loc[ind,:].copy()

    afternoon_ind = {
        'all':temp_big['rhours']<=25,
        'morning':(temp_big['rhours']<12) & (temp_big['ahours']<12),
        'afternoon':(temp_big['rhours']>=12)
    }
    k = 'all'; eqq_weighted = True
    for eqq_weighted in [True]:
        for k in afternoon_ind.keys():
            temp = temp_big.loc[afternoon_ind[k],:]
            # temp = temp.loc[temp['year']>2008,:]
            temp = temp.groupby(['dist_to_coverage', 'permno','date'])['ip'].sum().reset_index()
            temp = temp.groupby(['dist_to_coverage','permno'])['ip'].mean().reset_index()
            # temp = temmp
            temp

            if eqq_weighted:
                temp['ip']/=temp.groupby('permno')['ip'].transform('sum')
            m = pd.DataFrame(temp.groupby(['dist_to_coverage'])['ip'].mean())
            s = pd.DataFrame(temp.groupby(['dist_to_coverage'])['ip'].std())
            c = pd.DataFrame(temp.groupby(['dist_to_coverage'])['ip'].count())
            plot_ev(m,s,c,do_cumulate=False)
            plt.xlabel('Hours')
            plt.ylabel('# Logs')
            plt.tight_layout()
            plt.savefig(save_dir+f'logs_high_when_news_mean_{k}_equ{eqq_weighted}')
            plt.show()



    ## BUILD TABLE LOOKINGS AT ITEMS TYPE AND EVENT
    df['afternoon'] = df['ahours']>12
    temp = df.groupby(['afternoon','items'])['news0'].mean().reset_index().pivot(columns='afternoon',index='items',values='news0')
    use_itmes = Constant.LIST_ITEMS_TO_USE
    use_itmes.remove(6.01)
    temp['diff'] = temp[True]-temp[False]
    res_c = temp.loc[use_itmes,:]
    res_c['desc'] = res_c.index.map(Constant.ITEMS)
    res_c = res_c.sort_values('diff')

    table_to_latex_complying_with_attila_totally_unreasonable_demands(res_c,rnd=2,paths=save_dir,name='afternoon_table')


    temp = df.groupby(['afternoon','items'])['news0'].count().reset_index().pivot(columns='afternoon',index='items',values='news0')
    temp = (100*temp/(temp.sum(1).values.reshape(-1,1))).round(2)

    res_c = temp.loc[use_itmes,:]
    res_c['desc'] = res_c.index.map(Constant.ITEMS)
    res_c = res_c.sort_values(True)

    table_to_latex_complying_with_attila_totally_unreasonable_demands(res_c,rnd=2,paths=save_dir,name='afternoon_table_count')

    ####### looking at the percentage of high freq across time.
    df['first_hour'] =df['dist_to_form'] <=0
    df['first_covered_hours'] =df['dist_to_coverage'] <=0

    df['ym']= pd.to_datetime(didi.PandasPlus.get_ym(df['date']),format='%Y%m')
    c = 'first_hour'
    temp = df.loc[:,:].groupby(['ym',c])['ip'].sum().reset_index().pivot(columns=c,index='ym',values='ip')
    temp = temp.dropna().sort_index()
    temp = temp[True]/temp.sum(1)
    temp.plot()
    plt.show()

    temp =df.loc[df['date'].dt.year>2008,:].groupby('dist_to_form')['ip'].sum()
    (temp/temp.sum()).plot()
    plt.grid()
    plt.show()

    temp =df.loc[df['date'].dt.year<2008,:].groupby('dist_to_form')['ip'].sum()
    (temp/temp.sum()).plot()
    plt.grid()
    plt.show()


    temp  = df.groupby(['date','permno'])['dist_to_form'].sum()







