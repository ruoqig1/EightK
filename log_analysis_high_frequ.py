import didipack as didi
from matplotlib import pyplot as plt
from data import Data
from parameters import *
import seaborn as sns
# Function to convert to HH:MM:SS format
def convert_time(t):
    t_str = str(t).zfill(6)  # Convert to string and ensure 6 characters
    if pd.isna(t):
        return f"00:00:00"
    else:
        return f"{t_str[0:2]}:{t_str[2:4]}:{t_str[4:6]}"

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

    df['time'] = df['time'].apply(convert_time)
    df['atime'] = df['atime'].apply(convert_time)
    df['rtime'] = df['rtime'].apply(convert_time)

    df['time'] = pd.to_timedelta(df['time'],errors='coerce')
    df['rtime'] = pd.to_timedelta(df['rtime'],errors='coerce')
    df['atime'] = pd.to_timedelta(df['atime'],errors='coerce')

    # total number of covered arround covered date
    df['dist'] = (df['time']-df['rtime']).dt.total_seconds() / 3600
    ind = df['dist'].between(-24,24)
    temp = df.loc[ind,:]
    temp = temp.loc[df['news0']==1,:].copy()
    temp['dist'] = temp['dist'].round()
    temp = temp.groupby(['dist', 'permno','date'])['ip'].sum().reset_index()
    temp = temp.groupby(['dist','permno'])['ip'].mean().reset_index()
    temp.groupby(['dist'])['ip'].median().plot()
    plt.xlabel('Hours-Form Publication Hours')
    plt.ylabel('Mean Number of Views')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir+'logs_median_per_publication')
    plt.show()


    # percentage of downlaod for covered, arround covered time
    df['dist'] = (df['time']-df['rtime']).dt.total_seconds() / 3600
    ind = df['dist'].between(-24,24)
    temp = df.loc[ind,:]
    temp = temp.loc[df['news0']==1,:].copy()
    temp['dist'] = temp['dist'].round()
    temp = temp.groupby(['dist', 'permno','date'])['ip'].sum().reset_index()
    tot_down = temp.groupby(['permno','date'])['ip'].transform('sum')
    temp['ip']/=tot_down
    temp = temp.loc[tot_down>=100,:].groupby(['dist','permno'])['ip'].mean().reset_index()
    temp.groupby(['dist'])['ip'].mean().plot()
    plt.title(f'MCAP all')
    plt.grid()
    plt.xlabel('Hours-Form Publication Hours')
    plt.ylabel('Mean Percentage Of View')
    plt.tight_layout()
    plt.savefig(save_dir+'logs_perc_view_per_publication')
    plt.show()

    # failed first attempt by time
    df['covered_now'] = ((df['news0']==1) & (df['time']>=df['atime']))*1

    df['dist'] = (df['time']-df['atime']).dt.total_seconds() / 3600
    ind = df['dist'].between(-24,24)
    temp = df.loc[ind,:].copy()
    temp['type'] = 'uncovered'
    temp.loc[temp['news0']==1,'type']='to be covered'
    temp.loc[(temp['news0']==1) & (df['time']>=df['rtime']),'type']='covered now'

    temp['dist'] = temp['dist'].round()
    temp = temp.groupby(['dist', 'permno','date', 'type'])['ip'].sum().reset_index()
    temp = temp.groupby(['dist','permno','type'])['ip'].median().reset_index().pivot(columns='type',index=['dist','permno'],values='ip')
    temp =temp/temp[['uncovered']].values
    temp = temp.dropna().reset_index()
    temp =temp.drop(columns='permno').groupby('dist').mean()
    temp.plot()

    plt.title(f'MCAP all')
    plt.grid()
    plt.show()


    # run on the sample that has some coverage but all before coverage!
    df['dist'] = (df['time']-df['atime']).dt.total_seconds() / 3600
    ind = df['dist'].between(-24, 24)
    temp = df.loc[ind,:].copy()
    ind = ((temp['news0']==0)) | ((temp['news0']==1) & (df['time']<df['rtime']))
    temp = temp.loc[ind,:].copy()

    temp['dist'] = temp['dist'].round()
    # temp['dist'] = 1
    temp = temp.groupby(['dist', 'permno','date', 'news0'])['ip'].sum().reset_index()
    temp = temp.groupby(['dist','permno','news0'])['ip'].mean().reset_index().pivot(columns='news0',index=['dist','permno'],values='ip')

    hist_per_permno = temp.groupby('permno').mean()
    hist_per_permno['ratio'] = hist_per_permno[1.0]/hist_per_permno[0.0]
    hist_per_permno = hist_per_permno.dropna()
    hist_per_permno['ratio'].clip(-5,5).hist(bins=100)
    plt.grid()
    plt.xlabel('Clipped Ratio To Be Covered/Uncovered')
    plt.tight_layout()
    plt.savefig(save_dir+'logs_hist_to_be_covered')
    plt.show()
    plt.show()

    temp['ratio'] =temp[1.0]/temp[0.0]
    temp = temp.dropna().reset_index()
    temp.drop(columns='permno').groupby('dist')['ratio'].median().plot()
    plt.title(f'MCAP all')
    plt.grid()
    plt.xlabel('Hours-Form Publication Hours')
    plt.ylabel('Mean Ratio To Be Covered/Uncovered')
    plt.tight_layout()
    plt.savefig(save_dir+'logs_ts_to_be_covered')
    plt.show()


    # try to estimate percentage downloaded in the hours of, publciaiton, coverage
    df['dist_to_coverage'] = ((df['time'] - df['rtime']).dt.total_seconds() / 3600).round()
    df['dist_to_form'] = ((df['time'] - df['atime']).dt.total_seconds() / 3600).round()
    df['hour_type'] = 'normal'
    df.loc[df['dist_to_form']==0,'hour_type'] = 'Publication'
    df.loc[df['dist_to_coverage']==0,'hour_type'] = 'Coverage'
    df.loc[(df['dist_to_coverage']==0) & (df['dist_to_form']==0),'hour_type'] = 'Coverage & Publication'

    ind = df['dist_to_form'].between(-24, 24)
    temp = df.loc[ind, :]
    temp = temp.groupby(['hour_type','permno','date','news0'])['ip'].sum().reset_index()
    tot_down = temp.groupby(['permno','date','news0'])['ip'].transform('sum')
    # temp['ip']/= tot_down
    temp = temp.loc[tot_down>100,:].groupby(['news0','hour_type'])['ip'].sum().reset_index()
    temp['ip'] = temp['ip']/temp.groupby('news0')['ip'].transform('sum')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='news0', y='ip', hue='hour_type', data=temp)

    # Adding titles and labels
    plt.title('IP per News and Hour Type')
    plt.xlabel('News')
    plt.ylabel('IP')
    plt.xlabel('Uncovered/Covered')
    plt.ylabel('Percentage of Total Download')
    plt.tight_layout()
    plt.savefig(save_dir+'logs_bar')
    plt.show()
