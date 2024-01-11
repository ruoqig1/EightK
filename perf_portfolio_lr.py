import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
from didipack.trainer.train_splitter import get_start_dates, get_chunks, get_tasks_for_current_node
import psutil
from train_main import set_ids_to_eight_k_df
from experiments_params import get_main_experiments
from scipy import stats
from matplotlib import pyplot as plt
import regex as re
from tqdm import tqdm


def get_nth_day_return(row, n_days_ahead=1, permno_mapping=None, permno_date_to_index_mapping=None):
    """
    Given a row of the news dataframe, map it to the nth day head CRSP returns data
    """
    permno = row['permno']
    date_news = row['date_news']

    trading_days = permno_mapping.get(permno, None)
    date_to_index = permno_date_to_index_mapping.get(permno, {})

    if trading_days is not None and date_news in date_to_index:
        start_index = date_to_index[date_news]
        nth_index = start_index + n_days_ahead

        if nth_index < len(trading_days):
            nth_day_data = trading_days[nth_index]
            return pd.Series({
                'date': nth_day_data.date,
                'prc': nth_day_data.prc,
                'ret': nth_day_data.ret,
                'bid': nth_day_data.bid,
                'ask': nth_day_data.ask,
                'shrout': nth_day_data.shrout
            })
    # Return default values if conditions are not met
    return pd.Series({
        'date': pd.NaT,
        'prc': None,
        'ret': None,
        'bid': None,
        'ask': None,
        'shrout': None
    })


if __name__ == "__main__":
    args = didi.parse()
    par = Params()
    data = Data(par)
    load_dir = r'\data\gpfs\projects\punim2039\EightK\res\temp\vec_pred\logistic_regression\OPT_125m\NEWS_SINGLE'
    par.load(load_dir, f'/basic_parameters.p')

    # options
    n_days_ahead = 1
    threshold = 0.2

    model_outputs = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if re.match(r'\d+.p$', f)]
    par.load(load_dir, f'/basic_parameters.p')
    for model_output in model_outputs:
        df = pd.read_pickle(model_output).rename(columns={'ticker': 'permno'})
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={'date': 'date_news'}, inplace=True)

        crsp = data.load_crsp_daily()
        crsp_sorted = crsp.sort_values(by=['permno', 'date'])

        print('Creating permno mapping')
        crsp_grouped = crsp_sorted.groupby('permno')
        permno_mapping = {}
        permno_date_to_index_mapping = {}  # New mapping for date to index

        for permno, group in tqdm(crsp_grouped):
            records = group[['date', 'prc', 'ret', 'bid', 'ask', 'shrout']].to_records(index=False)
            permno_mapping[permno] = records
            permno_date_to_index_mapping[permno] = {date: idx for idx, date in enumerate(group['date'])}

        tqdm.pandas(desc="Processing rows")
        additional_columns = df.progress_apply(get_nth_day_return,
                                               axis=1,
                                               args=(n_days_ahead, permno_mapping, permno_date_to_index_mapping))
        for col in ['prc', 'ret', 'bid', 'ask', 'shrout']:
            additional_columns[col] = pd.to_numeric(additional_columns[col], errors='coerce')
        df_merged = pd.concat([df, additional_columns], axis=1)
        df_merged.rename(columns={'date': 'date_trade'}, inplace=True)

        # group by permno and date_news
        df_grouped = df_merged.groupby(['permno', 'date_news'])[
            ['ret', 'y_pred_prb', 'y_true', 'prc']].mean().reset_index()

        df_grouped['year'] = df_grouped['date_news'].dt.year
        print(df_grouped.groupby('year')['y_pred_prb'].mean())
        df_grouped['tresh'] = df_grouped.groupby('year')['y_pred_prb'].transform('mean')
        df_grouped['pred'] = df_grouped['y_pred_prb'] > df_grouped['tresh']
        # df['pred'] = df['y_pred_prb']>0
        df_grouped['accuracy'] = df_grouped['pred'] == df_grouped['y_true']
        df_grouped['accuracy'].mean()

        crsp['ret'] = pd.to_numeric(crsp['ret'], errors='coerce')
        df = df.merge(crsp[['date', 'ticker', 'permno', 'ret']])

        df = df.groupby(['permno', 'date'])[['ret', 'y_pred_prb', 'y_true']].mean().reset_index()

        df['year'] = df['date'].dt.year
        print(df.groupby('year')['y_pred_prb'].mean())
        df['tresh'] = df.groupby('year')['y_pred_prb'].transform('mean')

        df['pred'] = df['y_pred_prb'] > df['tresh']
        # df['pred'] = df['y_pred_prb']>0
        df['accuracy'] = df['pred'] == df['y_true']
        df['accuracy'].mean()

        df_grouped.groupby('date_news')['permno'].count().plot()
        df_grouped['pct'] = df_grouped.groupby('date_news')['y_pred_prb'].rank(pct=True)
        df_grouped.head()

        df_grouped['pos'] = 1 * (df_grouped['pct'] > (1 - threshold)) - (df_grouped['pct'] <= threshold) * 1
        ret = df_grouped.groupby(['date_news', 'pos'])['ret'].mean().reset_index().pivot(columns='pos',
                                                                                         index='date_news',
                                                                                         values='ret')
        ret[0] = ret[1] - ret[-1]
        ret.cumsum().plot()
        sh = np.sqrt(252) * (ret.mean() / ret.std()).round(3)
        plt.title(f'model_index {"LR"} sharpe: {np.round(sh[0], 3)}')
        plt.show()

        # index ret by [1, -1, 0]
        ret_formated = ret[[1, -1, 0]]
        sh_formated = np.sqrt(252) * (ret_formated.mean() / ret_formated.std()).round(3)
        print("        ", "Long", "Short", "L-S", sep="\t")
        print("Return: ", *list((252 * ret_formated.mean()).round(3)), sep="\t")
        print("Std:    ", *list((np.sqrt(252) * ret_formated.std()).round(3)), sep="\t")
        print("Sharpe: ", *list(sh_formated.round(3)), sep="\t")
