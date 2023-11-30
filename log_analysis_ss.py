import didipack as didi
from matplotlib import pyplot as plt
from data import Data
from parameters import *
from didipack import PandasPlus
from utils_local.plot import big_items_by_items_plots
from utils_local.general import table_to_latex_complying_with_attila_totally_unreasonable_demands



if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    save_dir = Constant.EMB_PAPER
    data = Data(par)
    df = data.load_logs_ev_study()

    ati = data.load_icf_ati_filter()
    ati = ati.rename(columns={'date':'form_date'})
    ati = ati.loc[ati['items'].isin(Constant.LIST_ITEMS_TO_USE),['form_date','form_id','news0','permno']].drop_duplicates()
    df = df.merge(ati)
    df['year'] = df['form_date'].dt.year
    df['ym'] = pd.to_datetime(PandasPlus.get_ym(df['form_date']),format='%Y%m')

    df.groupby('ym')['ip'].sum().plot()
    plt.ylabel('# of logs')
    plt.xlabel('Months')
    plt.tight_layout()
    plt.savefig(save_dir+'log_ss_nb_ts')
    plt.show()

    df = df.merge(data.load_mkt_cap_yearly())
    inds_todo = {
        'all': df['permno']>0,
        'Long Term Logs': df['dist']>40,
        'Short Term Logs': df['dist']<=40,
        'On The day Logs': df['dist']==0
    }
    i=0
    for k in inds_todo.keys():
        t=df.loc[inds_todo[k],:].groupby(['ym','permno','mcap_d'])['ip'].sum().reset_index().groupby(['ym','mcap_d'])['ip'].median().reset_index().pivot(columns='mcap_d',values='ip',index='ym')
        t.rolling(12).mean().plot()
        plt.ylabel('Mean # of logs (smooth yearly)')
        plt.xlabel('Months')
        plt.title(k)
        plt.tight_layout()
        plt.savefig(save_dir+f'log_ss_nb_ts_per_mcap_{i}.png')
        i+=1
        plt.show()




    # keeping items
    df = data.load_logs_ev_study()

    ati = data.load_icf_ati_filter()
    ati = ati.rename(columns={'date':'form_date'})
    ati = ati.loc[ati['items'].isin(Constant.LIST_ITEMS_TO_USE),['form_date','form_id','news0','permno','items']].drop_duplicates()
    df = df.merge(ati)
    df['year'] = df['form_date'].dt.year

    df = df.merge(data.load_mkt_cap_yearly())
    df['ym'] = pd.to_datetime(PandasPlus.get_ym(df['form_date']),format='%Y%m')
    inds_todo = {
        'all': df['permno']>0,
        'Long Term Logs': df['dist']>40,
        'Short Term Logs': df['dist']<=40,
        'On The day Logs': df['dist']==0
    }
    i = 0
    res = []
    res_c = []
    for k in inds_todo.keys():
        t=df.loc[inds_todo[k],:].groupby(['items','permno','mcap_d'])['ip'].sum().reset_index().groupby(['items','mcap_d'])['ip'].sum().reset_index().pivot(columns='mcap_d',values='ip',index='items')
        t=df.loc[inds_todo[k],:].groupby(['items','mcap_d'])['ip'].sum().reset_index().pivot(columns='mcap_d',values='ip',index='items').fillna(0.0)
        # t=(t/t.sum()).fillna(0.0)
        big_items_by_items_plots(t)
        plt.ylabel('Tot Number Logs')
        plt.title(k)
        plt.tight_layout()
        plt.savefig(save_dir+f'log_ss_items_mcap_{i}.png')
        i+=1
        plt.show()

        # most viewed events overall with description
        t = df.loc[inds_todo[k], :].groupby(['items'])['ip'].sum()
        c = df.loc[inds_todo[k], :].groupby(['items'])['form_id'].nunique()
        t.name = k
        res.append(t)
        c = t/c
        c.name = k
        res_c.append(c)


    res = pd.concat(res,axis=1).sort_values('On The day Logs',ascending=False)
    res = res.iloc[2:,:]
    res = (res/res.sum() *100).round(2)
    res['desc'] = res.index.map(Constant.ITEMS)
    table_to_latex_complying_with_attila_totally_unreasonable_demands(res,rnd=2,paths=save_dir,name='logs_ss_popular_forms')



    res_c = pd.concat(res_c,axis=1).sort_values('On The day Logs',ascending=False)
    res_c = (res_c/res_c.max()).round(2)
    res_c['desc'] = res_c.index.map(Constant.ITEMS)
    table_to_latex_complying_with_attila_totally_unreasonable_demands(res_c,rnd=2,paths=save_dir,name='logs_ss_popular_forms_normalise')