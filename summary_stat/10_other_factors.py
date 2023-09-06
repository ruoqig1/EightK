from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, OlsPLus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS


if __name__ == "__main__":
    par = Params()
    data = Data(par)
    reload =False

    save_dir = Constant.DROP_RES_DIR + f'/ss/10_other_factors/'
    os.makedirs(save_dir, exist_ok=True)
    with_correct_first_date =True

    df = pd.read_pickle(Constant.DROP_RES_DIR + f'/ss/05_no_coverage/df.p')
    df = df.loc[df['date'].dt.year>=2004,:]
    df['abs abret0'] = df['abret0'].abs()
    df =df.drop(columns=['mcap'])

    fund = data.load_fundamentals(True)

    df=df.merge(fund)
    ind=df['mcap_d']>=3
    df=df.loc[ind,:]

    # run the main table
    for fund_var in ['pe','roa','mom','mcap']:
    # for fund_var in ['mcap']:
        t = df.groupby('date')[fund_var].rank(pct=True)
        grp_dict = {}
        cutoff = 0.1
        grp_dict['Low'] = (t <=cutoff)
        grp_dict['Mid'] = (t > cutoff) & (t <= (1-cutoff))
        grp_dict['High'] = (t > (1-cutoff))
        table = didi.TableReg()
        for y in ['abvol0','abs abret0']:
            for grp in ['Low','Mid','High']:
                news_x = 'news0'
                eightk_x = 'eightk'
                iter = news_x+'*'+eightk_x
                df[iter] = df[news_x]*df[eightk_x]
                x_list = [news_x,eightk_x,iter]
                fe =['date','permno']
                olsPlus = OlsPLus()
                temp = df.copy().loc[grp_dict[grp],x_list+[y]+fe].dropna()

                l = 'Volume' if y =='abvol0' else 'Return'
                m=olsPlus.sm_with_fixed_effect(df=temp,y=y,x=x_list,fix_effect_cols=fe,std_error_cluster='date')
                table.add_reg(m,blocks=[{'Indep. Variable:':l, f'{fund_var.capitalize()}':str(grp)}, {'Time FE': 'True', 'Firm FE':'True'}])
            print(f'Done for y {y}')
        table.save_tex(save_dir=save_dir+f'fund_reg_{fund_var}.tex')
        print('Done for var')