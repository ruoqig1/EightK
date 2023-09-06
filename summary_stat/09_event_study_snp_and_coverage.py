from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    window = 40

    save_dir = Constant.DROP_RES_DIR + f'/ss/05_no_coverage/'
    os.makedirs(save_dir,exist_ok=True)

    df = data.load_some_relevance_icf()
    snp = data.load_snp_const()
    df['ym'] = PandasPlus.get_ym(df['adate'])
    snp['ym'] = PandasPlus.get_ym(snp['date'])
    snp = snp.drop(columns=['date'])

    df=df.merge(snp)
    df['in_snp'] = (((df['adate']<=df['ending'])  & (df['adate']>=df['start']) )*1).fillna(0.0)

    df.groupby('adate')['in_snp'].sum()

    df['evt'] = (df['adate']-df['start']).dt.days
    window = 60
    ind = (df['evt']>=-window) & (df['evt']<=window)
    temp=df.loc[ind,:].groupby(['evt','adate','permno'])[['no_rel','in_snp']].mean().reset_index()
    # temp['evt_m']=np.ceil(temp['evt']/20)
    temp.groupby('evt')['no_rel'].mean()
    smf.logit('no_rel ~ in_snp',temp).fit(cov_type='cluster', cov_kwds={'groups': temp['permno']}).summary()
    smf.logit('no_rel ~ in_snp',temp).fit().summary()

