
from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm
from didipack import PandasPlus
from statsmodels import api as sm
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from didipack import OlsPLus
from parameters import BRYAN_MAIN_CATEGORIES

def load_conditional_on_press_and_coerage(wsj = False,margin_in_perc = 1.25):
    if wsj:
        df = data.load_main_cosine('wsj_one_per_stock')
    else:
        df = pd.read_pickle('data/cosine/opt_model_typeOptModelType.BOW1news_sourceNewsSource.WSJ_ONE_PER_STOCKnb_chunks100save_chunks_size500chunk_to_run_id1/df.p')
    df = df.loc[df['news_prov']==1,:]
    df = df.rename(columns={'value':'cosine'})
    cos_firm = df.loc[df['dist'].abs()>20,:].groupby('permno')['cosine'].mean()
    cos_at = df.loc[df['dist'].abs() <=3, :].groupby('permno')['cosine'].mean()
    cos_at=cos_at.reset_index().rename(columns={'cosine':'cos_at'})
    cos_firm=cos_firm.reset_index().rename(columns={'cosine':'cos_m'})
    cos_firm = cos_firm.merge(cos_at)
    df = df.groupby(['form_date','dist','permno'])['cosine'].max().reset_index()

    # sanity check plot
    # df.groupby('dist')['cosine'].mean().plot()
    # plt.show()

    # select the coverage with highest cosine one day arround the event
    df = df.loc[df['dist'].between(-1,1),:].groupby(['form_date','permno']).max().reset_index()

    # ev = data.load_some_relevance_icf()
    # ev = ev.loc[ev['items'].isin(Constant.LIST_ITEMS_TO_USE),['items','adate','permno','atime']].rename(columns={'adate':'form_date'})
    #
    # df = ev.merge(df)
    df['cosine'] = df['cosine'].fillna(-1)

    df = df.merge(cos_firm)
    df['covered'] = 1*(df['cosine']>df['cos_m']*margin_in_perc)
    df['covered'].mean()
    bryan = data.load_bryan_data()
    bryan['ym'] = PandasPlus.get_ym(bryan['date'])
    df['ym'] = PandasPlus.get_ym(df['form_date'])
    pd.isna(bryan['market_equity']).mean()
    df = df.merge(bryan, left_on=['permno', 'ym'], right_on=['permno', 'ym'], how='left')
    return df

def load_coverage_as_in_first_paper(data):
    df = data.load_some_relevance_icf()
    df['covered'] = (df['relevance']>0)*1
    df['form_date'] = df['adate']
    d = pd.get_dummies(df['items'])
    df = pd.concat([df,d],axis=1)

    df = df[['form_date','permno','covered']+list(d.columns)].drop_duplicates()
    df['cosine'] =1
    bryan = data.load_bryan_data()
    bryan['ym'] = PandasPlus.get_ym(bryan['date'])
    df['ym'] = PandasPlus.get_ym(df['form_date'])
    pd.isna(bryan['market_equity']).mean()
    df = df.merge(bryan,left_on =['permno','ym'],right_on=['permno','ym'],how='left')
    cov = data.load_control_coverage().drop(columns='index')
    cov['permno'] = cov['permno'].astype(int)
    df=df.merge(cov,left_on = ['permno','ym'],right_on = ['permno','ym'],how='left')
    return df

def run_all_single_regressions(df):
    res = []
    control = ['market_equity','age','cov_pct_l']
    failed = []
    with_fe = False
    for cat in BRYAN_MAIN_CATEGORIES.keys():
        for variable in BRYAN_MAIN_CATEGORIES[cat]:
            try:
                temp = df[['cosine','covered','permno','form_date', variable]+control].dropna()
                if with_fe:
                    fe= pd.get_dummies(temp['items'])
                    const = list(fe.columns)+['const']
                    temp = pd.concat([temp,fe],axis=1)
                else:
                    const = ['const']
                temp['const']  =1.0
                for c in [variable]+control:
                    temp[c] = (temp[c]-temp[c].mean())/temp[c].std()

                m=sm.Logit(temp['covered'], temp[[variable]+control+const]).fit(cov_type='cluster', cov_kwds={'groups': temp['form_date']})
                res.append(pd.Series({'cat':cat,'var':variable, 'pseudo r':m.prsquared, 'params':m.params[variable], 'tstat':m.tvalues[variable]}))
            except:
                failed.append((cat,variable))

    res=pd.DataFrame(res)
    res['tstat_abs'] = res['tstat'].abs()
    res = res.dropna()
    res = res.sort_values('tstat_abs',ascending=False)
    print(res.round(3))
    #
    # print(res.groupby('cat')[['tstat_abs','params','pseudo r']].max().sort_values('tstat_abs'))
    # #
    for c in res['cat'].unique():
        print('#'*50)
        print(c)
        print('#'*50)
        print(res.loc[res['cat'] == c, :])
    return res


if __name__ == '__main__':
    par = Params()
    data = Data(par)
    # df = load_conditional_on_press_and_coerage(wsj=True,margin_in_perc=1.25)
    df = load_coverage_as_in_first_paper(data)


    variable = ['age','tangibility','lti_gr1a','dbnetis_at', 'at_turnover', #,'corr_1260d'
     'ni_inc8q','turnover_var_126d','dolvol_var_126d','zero_trades_252d',
     # 'betadown_252d','betabab_1260d','beta_60m','ivol_capm_252d',
     'beta_60m','ivol_capm_252d',
     'market_equity',
     # 'ret_9_1','ret_3_1',
     'ret_9_1',
     'eqnpo_12m','be_me','emp_gr1','col_gr1a',
     'ret_1_0'
     ]



    controls = ['cov_pct_l']
    # controls = []

    other_col = ['cosine', 'covered', 'permno', 'form_date','cov_pct_l']
    # other_col = ['cosine', 'covered', 'permno', 'form_date']

    temp = df[other_col+variable].dropna().copy()

    fe = pd.get_dummies(np.floor((temp['cov_pct_l']-0.00001)*100))
    fe.sum(1).max()
    temp = pd.concat([temp,fe],axis=1)
    controls = list(fe.columns)
    controls.remove(0.0)
    controls =controls


    const = ['const']
    const = []
    m_quantile = temp.groupby('form_date')['market_equity'].rank(pct=True)
    ind = m_quantile<1.01
    temp = temp.loc[ind,:]

    temp['market_equity'] = np.log(temp['market_equity'])
    temp['const'] = 1.0
    for v in variable:
        temp[v] = (temp[v] - temp[v].mean()) / temp[v].std()
    m = sm.Logit(temp['covered'], temp[variable + const+controls]).fit(cov_type='cluster', cov_kwds={'groups': temp['form_date']})

    print(m.summary())




