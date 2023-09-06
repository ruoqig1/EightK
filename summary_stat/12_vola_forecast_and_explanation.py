from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, OlsPLus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf


if __name__ == "__main__":
    par = Params()
    data = Data(par)

    olsPlus = OlsPLus()
    reload =False

    save_dir = Constant.DROP_RES_DIR + f'/ss/12_vola_forecast_and_explanation/'
    os.makedirs(save_dir, exist_ok=True)
    with_correct_first_date =True

    df = pd.read_pickle(Constant.DROP_RES_DIR + f'/ss/05_no_coverage/df.p')
    df = df.loc[df['date'].dt.year>=2005,:]
    df = df.merge(data.load_rav_avg_coverage())
    fund = data.load_fundamentals()

    df=df.drop(columns=['mcap'])
    df=df.merge(fund)
    df['ym'] = PandasPlus.get_ym(df['date'])


    fund = ['pe','roa','mom','mcap']
    fund_rank=[]
    for c in fund:
        n = f'{c}_r'
        df[n]=df.groupby('date')[c].rank(pct=True)
        fund_rank.append(n)

    fund.append('mcap_d')
    df['unc_eightk'] = ((df['eightk']==1) & (df['news0']==0))*1

    main_x = ['unc_eightk']
    control = ['news0','eightk','tot_news']
    control = ['eightk']
    x = fund_rank+control+main_x

    gp_date = 'ym'
    temp=df.groupby([gp_date,'permno'])[x+['mcap_d']].mean()
    temp['std'] = df.groupby([gp_date,'permno'])['ret'].std()
    temp = temp.reset_index()
    # temp = temp.loc[temp['eightk']>0,:]
    fe =[gp_date,'permno']

    temp['mcap_d']

    m = olsPlus.sm_with_fixed_effect(df=temp, y='std', x=x, fix_effect_cols=fe, std_error_cluster=gp_date)
    print(m.summary())
    df
    r=[]
    for mcap_d in np.sort(np.unique(temp['mcap_d'])):
        tt = temp.loc[temp['mcap_d']==mcap_d,:]
        print('#'*50,mcap_d)
        m = olsPlus.sm_with_fixed_effect(df=tt, y='std', x=x, fix_effect_cols=fe, std_error_cluster=gp_date)
        print(m.summary())
        r.append(pd.Series({'beta':m.params[main_x].iloc[0], 'p':m.pvalues[main_x].iloc[0]},name=mcap_d))

    pd.concat(r,axis=1).T
    # main_x=[]
    # for c in fund:
    #     n = f'{c}_r'
    #     df[n]=df.groupby('date')[c].rank(pct=True)
    #     main_x.append(n)
    #
    # x = main_x+['news','tot_news','vix']
    # x = main_x+['news']
    # y = ['news0']
    # std_error_cluster = 'date'
    # temp = df.loc[df['eightk']==1,y+x+[std_error_cluster]]
    # ym=pd.get_dummies(PandasPlus.get_ym(temp['date']))
    # ym_col = list(ym.columns)
    # temp = pd.concat([temp,ym*1],axis=1)
    #
    # temp['const']=1.0
    # print(sm.Logit(temp[y],temp[x+['const']]).fit(cov_type='cluster',cov_kwds={'groups':temp[std_error_cluster]}).summary2())
    # print(sm.Logit(temp[y],temp[x+ym_col]).fit(cov_type='cluster',cov_kwds={'groups':temp[std_error_cluster]}).summary2())
    # temp[x + ym_col].dtypes


