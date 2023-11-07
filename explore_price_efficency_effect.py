import pandas as pd
import tensorflow as tf
import os
from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm
from didipack import PandasPlus

from parameters import *
from data import Data
from matplotlib import pyplot as plt
from didipack import OlsPLus


def get_q_number_df(df):
    temp = df.loc[df['items'] == 2.02, ['date', 'permno']].rename(columns={'date': 'd_end'})
    temp['d_start'] = temp['d_end'] - pd.DateOffset(months=3)
    temp = temp.sort_values(['permno', 'd_start']).reset_index(drop=True)

    # Add a 'group_id' to the original DataFrame
    temp['group_id'] = temp.groupby('permno').cumcount() + 1

    # Create the boundaries for each 'permno'
    boundaries = temp.groupby('permno').agg({'d_start': 'min', 'd_end': 'max'}).reset_index()

    # Create a DataFrame with all possible days for each 'permno'
    all_dates_list = []
    for _, row in boundaries.iterrows():
        all_dates_list.append(pd.DataFrame({
            'date': pd.date_range(row['d_start'], row['d_end']),
            'permno': row['permno']
        }))

    all_dates_df = pd.concat(all_dates_list).reset_index(drop=True)

    # Merge the original DataFrame with the new DataFrame
    merged_df = pd.merge_asof(all_dates_df.sort_values('date'), temp.sort_values('d_start'), by='permno', left_on='date', right_on='d_start', direction='forward')

    # Drop unnecessary columns and carry the 'group_id' over as 'number'
    final_df = merged_df.drop(['d_start', 'd_end'], axis=1).rename(columns={'group_id': 'q_number'})
    final_df = final_df.dropna().sort_values(['permno', 'date']).reset_index(drop=True)
    print('build quarter df')
    return final_df
if __name__ == '__main__':
    par = Params()
    data = Data(par)

    df = data.load_some_relevance_icf()
    ev = data.load_abn_return()
    ret_col = ['sigma_ra', 'sigma_ret_train', 'sigma_abs_ra', 'abs_abret']
    ev=ev.loc[ev['evttime'].between(-1,1),:].groupby(['date','permno'])[ret_col].mean().reset_index()

    df['date'] = df['adate']
    df = df.merge(ev,how='left')
    q_df = get_q_number_df(df)

    df = df.merge(q_df)
    df['coverage'] = df['no_rel'] == 0



    nb_form = df.groupby(['q_number','permno'])[['form_id']].nunique().reset_index()
    nb_items = df.groupby(['q_number','permno'])[['items']].count().reset_index()
    coverage = df.groupby(['q_number','permno'])[['coverage']].mean().reset_index()
    ret = df.loc[df['items']==2.02,:].groupby(['q_number','permno'])[ret_col].last().reset_index()
    time = df.groupby(['q_number','permno'])['date'].max().reset_index()
    firm = df.groupby(['q_number','permno'])[['mcap_d','mcap']].last().reset_index()

    df = ret.merge(firm).merge(coverage).merge(nb_items).merge(nb_form).merge(time)
    #
    # df['big'] =df['mcap_d']==10
    # df['y'] = df['sigma_abs_ra']#/df['sigma_abs_ra']
    # df['y'] = df['abs_abret']#/df['sigma_abs_ra']
    df['y'] = df['abs_abret']/df['sigma_ra']
    # control = 'big'
    # df['x'] = np.round(df['coverage'].rank(pct=True)*10)
    # df['x'] = np.round(df['form_id'].rank(pct=True)*10)
    # df['x'] = np.round(df['coverage'].rank(pct=True)*3)
    # df.groupby('x').count()
    # df.groupby('x')['y'].mean().plot()
    # plt.show()
    #
    # df.groupby(['x',control])['y'].mean().reset_index().pivot(columns=control,index='x',values='y').plot()
    # plt.show()

    df['one'] = 1.0
    df['log_form'] = np.log(1+df['form_id'])
    df['year'] = df['date'].dt.year
    df['ym'] = PandasPlus.get_ym(df['date'])
    df['full_coverage'] = 1*(df['coverage']==1)
    ols = OlsPLus()
    ols.sm_with_fixed_effect(df=df,y='abs_abret',x=['coverage','form_id','sigma_abs_ra','mcap_d','one'],std_error_cluster='permno').summary()
    ols.sm_with_fixed_effect(df=df,y='abs_abret',x=['coverage','form_id','sigma_abs_ra','mcap_d'],fix_effect_cols=['year'],std_error_cluster='ym').summary()

    ols.sm_with_fixed_effect(df=df,y='abs_abret',x=['coverage','form_id'],fix_effect_cols=['year','permno'],std_error_cluster='ym').summary()

