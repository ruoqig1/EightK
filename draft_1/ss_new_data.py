import gc
from data import Data
import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from itertools import chain
from utils_local.plot import plot_ev, plot_ev_no_conf
from matplotlib import pyplot as plt
from didipack import PlotPlus, PandasPlus
import didipack as didi

def check_time_effect():
    ev = data.load_abn_return(1)
    ev['ym'] = PandasPlus.get_ym(ev['date'])
    ts  ='ym'
    temp = ev.loc[ev['evttime'].between(-1,1),:].groupby(['date','permno'])['abs_abret'].mean().reset_index()
    temp['year'] = temp['date'].dt.year
    temp['ym'] = PandasPlus.get_ym(temp['date'])
    high = temp.groupby(ts)['abs_abret'].mean()
    temp = ev.loc[~ev['evttime'].between(-1,1),:].groupby(['date','permno'])['abs_abret'].mean().reset_index()
    temp['year'] = temp['date'].dt.year
    temp['ym'] = PandasPlus.get_ym(temp['date'])
    low = temp.groupby(ts)['abs_abret'].mean()


    m = ((high-low)/low).reset_index()
    # m = ((high/low)).reset_index()
    temp = df.groupby(ts)['news'].mean()
    m['coverage'] = temp.values
    m=m.set_index(ts)
    m.index = pd.to_datetime(m.index,format='%Y%m')
    PlotPlus.plot_dual_axis(m,'abs_abret','coverage')
    plt.show()


def table_to_latex_complying_with_attila_totally_unreasonable_demands(df, rnd, paths, name):
    latex_str = df.round(rnd).to_latex(float_format=f"%.{rnd}f")

    lr = latex_str.split('tabular}{')[1].split('}')[0]
    latex_str = latex_str.replace(lr,lr[1:]+'}')
    latex_str=latex_str.replace(r'\begin{tabular}',r'\begin{tabular*}{1\linewidth}{@{\hskip\tabcolsep\extracolsep\fill}l*{1}')
    latex_str=latex_str.replace(r'\end{tabular}',r'\end{tabular*}')

    # Save the modified string to a file
    file_path = paths + name
    with open(file_path, 'w') as f:
        f.write(latex_str)



def do_one_mona_lisa(df,to_label,label_begining ='', color='Blues'):
    cmap = plt.get_cmap(color)

    list_id = np.linspace(0.25, 1, df.shape[1])
    for i, c in enumerate(df.columns):
        if c in to_label:
            plt.plot(df.index, df[c], color=cmap(list_id[i]), label=f'{label_begining}{c}')
        else:
            plt.plot(df.index, df[c], color=cmap(list_id[i]))


if __name__ == "__main__":
    args = didi.parse()
    par=Params()
    data = Data(par)
    use_constant_in_abret = False
    save_dir = Constant.EMB_PAPER+'ss/'

    os.makedirs(save_dir, exist_ok=True)


    df = data.load_some_relevance_icf()
    df['news'] = df['no_rel']==0
    df=df.rename(columns={'adate':'date'})

    df['year'] = df['date'].dt.year
    df['ym'] = PandasPlus.get_ym(df['date'])
    df.groupby(['year','permno'])['form_id'].nunique().reset_index().groupby('year')['form_id'].mean().plot()
    plt.show()

    temp = df.groupby(['year','permno','mcap_d'])['form_id'].nunique().reset_index().groupby(['year','mcap_d'])['form_id'].mean().reset_index().pivot(columns='mcap_d',index='year',values='form_id')
    do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    plt.show()

    temp = df.groupby(['year','permno','mcap_d'])['news'].mean().reset_index().groupby(['year','mcap_d'])['news'].mean().reset_index().pivot(columns='mcap_d',index='year',values='news')
    do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    plt.show()

    rav = data.load_ravenpack_all()
    rav['year'] = rav['rdate'].dt.year
    rav['ym'] = PandasPlus.get_ym(rav['rdate'])
    temp = rav.groupby('ym')['permno'].count()
    temp.plot()
    plt.show()



    temp=pd.DataFrame(temp)
    temp['coverage'] = df.groupby('ym')['news'].mean()
    temp = temp.dropna()
    temp.index = pd.to_datetime(temp.index,format='%Y%m')
    temp = temp.rolling(12).mean()
    PlotPlus.plot_dual_axis(temp,'permno','coverage')
    plt.show()
    temp.corr()

    # do_one_mona_lisa(temp.iloc[1:,:],to_label=[1,10],label_begining='MCAP Quantile: ')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_dir+'ss_nb_8k_per_items.png')
    # plt.show()

    coverage = df[['permno','mcap_d','news','form_id']].drop_duplicates().groupby(['permno','mcap_d'])['news'].mean().reset_index().groupby('mcap_d')['news'].mean()
    release = data.get_press_release_bool_per_event().merge(df[['cik','form_id','permno']].drop_duplicates()).merge(data.load_mkt_cap_yearly())
    release = release.groupby(['permno','mcap_d'])['release'].mean().reset_index().groupby('mcap_d')['release'].mean()
    nb_form_per_year = df[['permno', 'mcap_d','year', 'news', 'form_id']].drop_duplicates().groupby(['permno', 'mcap_d','year'])['news'].count().reset_index().groupby('mcap_d')['news'].mean()
    nb_items_per_year = df.groupby(['permno', 'mcap_d','year'])['news'].count().reset_index().groupby('mcap_d')['news'].mean()

    ev = data.load_abn_return(6)
    ev = ev.loc[ev['evttime'].isin([0]), ['abret', 'abs_abret', 'permno', 'date']]
    df = df.merge(ev,how='left')
    av_ret = df.groupby('mcap_d')['abret'].mean()*100
    av_ret_covered = df.loc[df['news']==1,:].groupby('mcap_d')['abret'].mean()*100
    av_ret_uncovered =df.loc[df['news']==0,:].groupby('mcap_d')['abret'].mean()*100

    av_absret = df.groupby('mcap_d')['abs_abret'].mean()*100
    av_absret_covered = df.loc[df['news']==1,:].groupby('mcap_d')['abs_abret'].mean()*100
    av_absret_uncovered =df.loc[df['news']==0,:].groupby('mcap_d')['abs_abret'].mean()*100

    r = rav.loc[rav['relevance']>0,:].groupby(['year','permno'])['rdate'].nunique().reset_index()
    r['permno'] = r['permno'].astype(int)
    perc_day_with_news = r.merge(data.load_mkt_cap_yearly()).groupby('mcap_d')['rdate'].mean()/365

    res = pd.DataFrame(coverage*100).rename(columns={'news':'8K w/Media Coverage (%)'})
    res['8K w/PR (%)'] = release*100
    res['Days w/Media Coverage (%)'] = perc_day_with_news*100
    res['# 8K-Forms per year'] = nb_form_per_year
    res['# 8K-Items per year'] = nb_items_per_year
    res['avg. ret (all)'] = av_ret
    res['avg. ret (covered)'] = av_ret_covered
    res['avg. ret (un-covered)'] = av_ret_uncovered
    res['avg. abs(ret) (all)'] = av_absret
    res['avg. abs(ret) (covered)'] = av_absret_covered
    res['avg. abs(ret) (un-covered)'] = av_absret_uncovered
    mcap_latex_name = 'MKT CAP (decile)'
    res.index.name = mcap_latex_name
    # res.T.round(2).to_latex(save_dir+'main_ss_table.tex', float_format="%.2f")
    table_to_latex_complying_with_attila_totally_unreasonable_demands(res.T.round(2), 2, save_dir, 'main_ss_table.tex')

    nb_per_year_per_items = df.groupby(['permno', 'mcap_d', 'year', 'items'])['news'].count().reset_index().groupby(['mcap_d', 'items'])['news'].mean().reset_index().pivot(columns='items', index='mcap_d', values='news').fillna(0.0).T
    nb_per_year_per_items.columns.name = mcap_latex_name
    # nb_per_year_per_items.round(2).to_latex(save_dir+'nb_of_items_per_year.tex', float_format="%.2f")
    table_to_latex_complying_with_attila_totally_unreasonable_demands(nb_per_year_per_items, 2, save_dir, 'nb_of_items_per_year.tex')


    # percentage of firms with at least one items per items
    a = (df.groupby(['permno', 'mcap_d','year','items'])['news'].count()>0).reset_index().groupby(['year','items','mcap_d'])['news'].sum().reset_index().rename(columns={'news':'nb_items'})
    b = df.groupby(['year','mcap_d'])['permno'].nunique().reset_index().rename(columns={'permno':'nb_firm'})
    nb= a.merge(b)
    nb['perc'] = nb['nb_items']/nb['nb_firm']
    nb = nb.groupby(['items','mcap_d'])['perc'].mean().reset_index().pivot(columns='items',index='mcap_d',values='perc').fillna(0.0).T

    nb.columns.name = mcap_latex_name
    nb *=100
    # nb.round(2).to_latex(save_dir+'perc_of_firm_with_items.tex', float_format="%.2f")
    table_to_latex_complying_with_attila_totally_unreasonable_demands(nb.round(2), 2, save_dir, 'perc_of_firm_with_items.tex')

    print('ran')
