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

    save_dir = Constant.EMB_PAPER + f'/res/ss/05_no_coverage/'
    os.makedirs(save_dir, exist_ok=True)
    with_correct_first_date =True

    if reload:
        window = 40

        df = data.load_abret_and_abvol()

        rav = data.load_ravenpack_all()
        ev = data.load_list_by_date_time_permno_type()
        if with_correct_first_date:
            ind=ev['first_date']>=ev['adate']
            ev = ev.loc[ind,:]
        # ev = ev.loc[ev['items']!=2.02]
        #
        # drop news and eightk before 16
        ind=rav['rtime'].apply(lambda x: int(x[:2])<=16)
        rav = rav.loc[ind,:]

        ind = pd.to_numeric(ev['atime'].apply(lambda x: str(x)[:2]),errors='coerce')<=16
        ev = ev.loc[ind,:]

        rav['news0'] = (rav['relevance']>=1)*1
        rav['news1'] = (rav['relevance']>=80)*1
        rav['news2'] = (rav['relevance']>=9)*1

        tot_news = rav.groupby('rdate')['news1'].count()
        tot_news = tot_news.reset_index().rename(columns={'news1':'tot_news','rdate':'date'})


        rav = rav.groupby(['rdate','permno'])[['news0','news1','news2']].max().reset_index()
        rav = rav.rename(columns={'rdate':'date'})
        rav['permno'] = rav['permno'].astype(int)

        ev=ev.groupby(['adate','permno'])['items'].count().reset_index().rename(columns={'items':'eightk','adate':'date'})
        ev['permno'] = ev['permno'].astype(int)
        ev['eightk']=ev['eightk']/ev['eightk']
        ev['date'] = pd.to_datetime(ev['date'])

        df=df.dropna()
        df=df.merge(ev,how='left').merge(rav,how='left')
        df =df.fillna(0.0)

        vix = data.load_yahoo(ticker='vix',freq='D')/100
        df = df.merge(vix.reset_index(),how='left')

        df['one'] = 1

        # add snp 500 id
        sp = data.load_snp_const(False)
        sp['ym'] = PandasPlus.get_ym(sp['date'])
        sp = sp.drop(columns='date')
        df['ym'] = PandasPlus.get_ym(df['date'])
        df = df.merge(sp,how='left')
        df['in_snp'] = (((df['date']<=df['ending'])  & (df['date']>=df['start']) )*1).fillna(0.0)

        ## add tot_news
        tot_news['tot_news'] = (tot_news['tot_news']-tot_news['tot_news'].mean())/tot_news['tot_news'].std()
        df = df.merge(tot_news,how='left')

        mcap = data.load_mkt_cap_yearly()
        df['year'] = df['date'].dt.year
        df = df.merge(mcap)
        # for reload
        if with_correct_first_date:
            df.to_pickle(save_dir+'df_with_correct_first_date.p')
        else:
            df.to_pickle(save_dir+'df.p')
    else:
        if with_correct_first_date:
            df = pd.read_pickle(save_dir+'df.p')
        else:
            df = pd.read_pickle(save_dir+'df_with_correct_first_date.p')


    df = df.loc[df['date'].dt.year>=2004,:]
    df['abs abret0'] = df['abret0'].abs()

    t=df.groupby('date')['in_snp'].sum()
    plt.show()

    # run the main table
    table = didi.TableReg()
    for y in ['abvol0','abs abret0']:
        for snp in [0,1]:
            ind = df['in_snp'] == snp
            news_x = 'news0'
            eightk_x = 'eightk'
            iter = news_x+'*'+eightk_x
            df[iter] = df[news_x]*df[eightk_x]
            x_list = [news_x,eightk_x,iter]
            fe =['date','permno']
            olsPlus = OlsPLus()
            temp = df.copy()
            l = 'Volume' if y =='abvol0' else 'Return'
            m=olsPlus.sm_with_fixed_effect(df=df.loc[ind,:],y=y,x=x_list,fix_effect_cols=fe,std_error_cluster='date')
            table.add_reg(m,blocks=[{'Indep. Variable:':l, 'Frims in S\&P500':str(snp==1)}, {'Time FE': 'True', 'Firm FE':'True'}])
        print(f'Done for y {y}')

    table.save_tex(save_dir=save_dir+'main_reg.tex')

    # run
    for with_news in [1,0,-1]:
        for y in ['abvol0','abs abret0']:
            table = didi.TableReg()
            for mcap in np.sort(np.unique(df['mcap_d'])):
                ind = (df['mcap_d'] == mcap)
                news_x = 'news0'
                eightk_x = 'eightk'
                iter = news_x+'*'+eightk_x
                df[iter] = df[news_x]*df[eightk_x]
                if with_news==1:
                    x_list = [news_x,eightk_x,iter]
                if with_news==0:
                    x_list = [eightk_x,iter]
                if with_news==-1:
                    x_list = [eightk_x]
                fe =['date','permno']
                olsPlus = OlsPLus()
                temp = df.copy()
                l = 'Volume' if y =='abvol0' else 'Return'
                m=olsPlus.sm_with_fixed_effect(df=df.loc[ind,:],y=y,x=x_list,fix_effect_cols=fe,std_error_cluster='date')
                table.add_reg(m,blocks=[{'Indep. Variable:':l, 'MCAP Decile':str(mcap)}, {'Time FE': 'True', 'Firm FE':'True'}])
                print('#'*50)
                print('Done for',mcap,y)
                print(m.summary2())
            if with_news==1:
                table.save_tex(save_dir=save_dir+f'{l}_mcap{mcap}.tex')
            if with_news==0:
                table.save_tex(save_dir=save_dir+f'no_news_{l}_mcap{mcap}.tex')
            if with_news==-1:
                table.save_tex(save_dir=save_dir+f'eight_only_{l}_mcap{mcap}.tex')










