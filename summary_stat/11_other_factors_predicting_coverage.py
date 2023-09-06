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
    reload =False

    save_dir = Constant.DROP_RES_DIR + f'/ss/11_other_factors_pred_coverage/'
    os.makedirs(save_dir, exist_ok=True)
    with_correct_first_date =True

    df = pd.read_pickle(Constant.DROP_RES_DIR + f'/ss/05_no_coverage/df.p')
    df = df.loc[df['date'].dt.year>=2005,:]
    df = df.merge(data.load_rav_avg_coverage())
    fund = data.load_fundamentals()

    df=df.drop(columns=['mcap'])
    df=df.merge(fund)


    fund = ['pe','roa','mom','mcap']
    main_x=[]
    for c in fund:
        n = f'{c}_r'
        df[n]=df.groupby('date')[c].rank(pct=True)
        main_x.append(n)

    x = main_x+['news','tot_news','vix']
    x = main_x+['news']
    y = ['news0']
    std_error_cluster = 'date'
    temp = df.loc[df['eightk']==1,y+x+[std_error_cluster]]
    ym=pd.get_dummies(PandasPlus.get_ym(temp['date']))
    ym_col = list(ym.columns)
    temp = pd.concat([temp,ym*1],axis=1)

    temp['const']=1.0
    print(sm.Logit(temp[y],temp[x+['const']]).fit(cov_type='cluster',cov_kwds={'groups':temp[std_error_cluster]}).summary2())
    print(sm.Logit(temp[y],temp[x+ym_col]).fit(cov_type='cluster',cov_kwds={'groups':temp[std_error_cluster]}).summary2())
    temp[x + ym_col].dtypes