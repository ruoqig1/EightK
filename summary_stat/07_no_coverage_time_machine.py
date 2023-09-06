from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus
from scipy.stats import ttest_ind
from utils_local.plot import *
from statsmodels import api as sm




if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/ss/05_no_coverage/'
    os.makedirs(save_dir,exist_ok=True)

    df = data.load_main_no_rel_data()

    df = df.loc[df['evttime']==0,:]
    df['ym'] = PandasPlus.get_ym(df['adate'])

    d=df.groupby(['year','no_rel'])['abret_abs'].mean().reset_index().pivot(columns='year',values='abret_abs',index='no_rel').T
    d.plot()
    plt.show()

    vix=data.load_yahoo('vix','D',False)
    df=df.merge(vix.reset_index())
    df['no_rel']
    df.groupby(['no_rel'])['relevance'].min()
    df['no_rel']= (df['relevance']<80)*1

    crsp = data.load_crsp_daily()
    crsp['ym'] = PandasPlus.get_yw(crsp['date'])
    crsp=crsp.groupby('ym')['permno'].nunique().reset_index()
    crsp = crsp.rename(columns={'permno':'nb_firm'})

    d=df.groupby(['year','ym','no_rel','mcap_d'])[['abret_abs','vix']].mean().reset_index()
    d=d.merge(crsp)
    year=(pd.get_dummies(d['year'])*1).reset_index(drop=True)
    mcap=(pd.get_dummies(d['mcap_d'])*1).reset_index(drop=True)
    mcap.columns = [f'mcap_d{x}' for x in mcap.columns]
    d = pd.concat([d,year,mcap],axis=1)

    ts_col = []
    for c in year:
        n = f'{c}*no_rel'
        ts_col.append(n)
        d[n] = d['no_rel']*d[c]

    d['nb_firm_log'] = np.log(d['nb_firm'])
    x_col =ts_col+ ['vix','nb_firm_log']#+list(year.columns)
    m =sm.OLS(d['abret_abs'],d[x_col]).fit()
    m.summary2()
    v=m.tvalues[ts_col]
    v.index = year.columns
    v.plot()
    plot_with_every_ticks_tight_and_grid(v.index)
    plt.show()
    print(m.summary2())


    p = {}
    t = {}
    for y in year.columns:
        dd = d.loc[d['year']==y,:]
        # dd = dd.loc[dd['mcap_d']<=9,:]
        x_col = ['no_rel', 'vix','nb_firm'] + list(mcap.columns)
        m = sm.OLS(dd['abret_abs'], dd[x_col]).fit()
        p[y] = m.params['no_rel']
        t[y] = m.tvalues['no_rel']

    p=pd.Series(t)
    p.plot()
    plot_with_every_ticks_tight_and_grid(p.index)
    plt.show()