import didipack as didi
import pandas as pd


import pandas as pd
import datetime
from matplotlib import pyplot as plt
from data import Data
from parameters import *
import seaborn as sns
from utils_local.plot import plot_ev
from utils_local.general import table_to_latex_complying_with_attila_totally_unreasonable_demands
import pandas as pd
import pytz
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
    df['rhours_min'] = df['rhours']+df['rmins']/60
    df['hours_min'] = df['hours']+df['mins']/60

    temp = pd.DataFrame(df.loc[ind,:].groupby('rhours_min')['ip'].count()).rename(columns={'ip':'% Coverage'})
    temp['% of 8k Forms (covered)'] = df.loc[ind,:].groupby('hours_min')['ip'].count()
    temp['% of 8k Forms (un-covered)'] = df.loc[ind_uncovered,:].groupby('hours_min')['ip'].count()
    # temp['% of logs'] = df.loc[ind_uncovered,:].groupby('hours')['ip'].sum()
    temp = (temp/temp.sum()).cumsum()

    temp.plot(figsize=(6.4 * 2, 4.8),label = 'Market Hours')
    plt.axvspan(9.5, 16, color='grey', alpha=0.5)
    plt.xticks(range(int(6), int(df['rhours'].max()) + 1))
    plt.xlabel('Hours')
    plt.ylabel('CDF')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir+'logs_high_hours_of_day_ss.png')
    plt.show()



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


def get_utc_offset(date):
    """
    Returns the UTC offset (5 or 4) for a given date in New York timezone.

    :param date: A string representing a date in 'YYYY-MM-DD' format.
    :return: An integer, either 5 or 4, representing the UTC offset.
    """
    ny_timezone = pytz.timezone('America/New_York')
    parsed_date = pd.to_datetime(date)

    # localize the date to New York timezone
    localized_date = ny_timezone.localize(parsed_date)

    # Check if the date is in DST
    if localized_date.dst() != datetime.timedelta(0):
        return 400  # EDT (UTC -4)
    else:
        return 500  # EST (UTC -5)


if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    data = Data(par)
    save_dir = Constant.EMB_PAPER
    pre_process = False
    if pre_process:
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
        # when zone is  500 or 400 it means the origina ldata had the correct time zone we need to correct the others
        # to do soe we start by getting the correction for all dates
        bug['utc_offset'] = bug['date'].apply(get_utc_offset)
        # then set it to zero when zone is not zero
        bug.loc[bug['zone']>0,'utc_offset']= 0

        # we then merge date and the utc offset
        df = df.merge(bug[['date','utc_offset']])
        df['utc_offset']/=100
        df['utc_offset'] = df['utc_offset'].astype(int)
        #
        df['time'] = df['time'].apply(convert_time)
        df['rtime'] = df['rtime'].apply(convert_time)
        # df['atime'] = df.groupby(['form_id', 'permno'])['time'].transform('min')

        df['time'] = pd.to_timedelta(df['time'],errors='coerce')
        df['rtime'] = pd.to_timedelta(df['rtime'],errors='coerce')
        df['atime'] = pd.to_timedelta(df['atime'],errors='coerce')
        # here at last we do the utc correction of the bugged dates
        df['time'] = df['time'] - pd.to_timedelta(df['utc_offset'], unit='h')
        df['year'] = df['date'].dt.year


        # total number of covered arround covered date
        df['dist_to_coverage_min'] = ((df['time'] - df['rtime']).dt.total_seconds() / 60).round()
        df['dist_to_form_min'] = ((df['time'] - df['atime']).dt.total_seconds() / 60).round()
        df['dist_news_to_a_min'] = ((df['rtime'] - df['atime']).dt.total_seconds() / 60).round()

        df['dist_to_coverage'] = (df['dist_to_coverage_min']/60).round()
        df['dist_to_form'] = (df['dist_to_form_min']/60).round()
        df['dist_news_to_a'] = (df['dist_news_to_a_min']/60).round()

        df['rhours']=df['rtime'].dt.components['hours']
        df['ahours']=df['atime'].dt.components['hours']
        df['hours']=df['time'].dt.components['hours']



        df['rmins']=df['rtime'].dt.components['minutes']
        df['amins']=df['atime'].dt.components['minutes']
        df['mins']=df['time'].dt.components['minutes']


        df.to_pickle(data.p_dir+'high_frequ_pre_process.p')
    else:
        df = pd.read_pickle(data.p_dir+'high_frequ_pre_process.p')

    plot_hours(df)
    df[['date','rtime']].dtypes

    ## BIG PLOTS ON NUMBER OF LOGS WHEN COVERAGE ARRIVE (removing the bump of new info
    ind = df['dist_to_coverage'].between(-12,12)
    temp = df.loc[ind,:]
    temp = temp.loc[df['news0']==1,:].copy()
    temp_big = temp.copy()
    # ind = (temp['dist_to_form']>0)& (temp['dist_news_to_a']>=0)
    # ind = (temp['dist_to_form']>0) & (temp['dist_news_to_a']>0)
    temp_big = temp.loc[ind,:].copy()

    afternoon_ind = {
        'all':temp_big['rhours']<=25,
        # 'morning':(temp_big['rhours']<12) & (temp_big['ahours']<12),
        # 'afternoon':(temp_big['rhours']>=12)
    }
    k = 'all'; eqq_weighted = True
    for eqq_weighted in [True, False]:
        for k in afternoon_ind.keys():
            for frequency_in_min in [True, False]:
                temp = temp_big.loc[afternoon_ind[k],:]
                dist_type = 'dist_to_coverage_min' if frequency_in_min else 'dist_to_coverage'
                temp = temp.groupby([dist_type, 'permno','date'])['ip'].sum().reset_index()
                temp = temp.groupby([dist_type,'permno'])['ip'].mean().reset_index()
                if frequency_in_min:
                    temp = temp.loc[temp[dist_type].between(-60,60),:]
                else:
                    temp = temp.loc[temp[dist_type].between(-6,6),:]
                if eqq_weighted:
                    temp['ip']/=temp.groupby(['permno'])['ip'].transform('sum')
                m = pd.DataFrame(temp.groupby([dist_type])['ip'].mean())
                s = pd.DataFrame(temp.groupby([dist_type])['ip'].std())
                c = pd.DataFrame(temp.groupby([dist_type])['ip'].count())
                plot_ev(m,s,c,do_cumulate=False)
                if frequency_in_min:
                    plt.xlabel('Minutes')
                else:
                    plt.xlabel('Hours')
                if eqq_weighted:
                    plt.ylabel('% Logs')
                else:
                    plt.ylabel('# Logs')
                plt.tight_layout()
                s = save_dir+f'logs_equ{eqq_weighted}_min{frequency_in_min}.png'
                plt.savefig(s)
                print(s)
                plt.title(f'{k}')
                plt.tight_layout()
                plt.show()





