import os

import numpy as np
import  tensorflow as tf
import pandas as pd
from parameters import *
from data import *
from train_tf import PipelineTrainer
from vec_to_tf_records import serialize_news_link_all_single_news
from experiments_params import get_main_experiments
from sklearn.linear_model import LogisticRegression

def gen_data():
    np.random.seed(1234)
    N = 10000
    P = 5120
    x = np.random.normal(size=(N, P))
    beta = np.random.normal(size=(P,),scale=0.1)
    sigma = 10.0
    y = x @ beta + np.random.normal(size=(N,), scale=sigma)
    df = pd.DataFrame()
    df['ret'] = y
    df['permno'] = 5
    df['ret_m'] = y
    df['reuters'] = 1
    df['alert'] = False
    df['id'] = [f'id_{x}' for x in range(N)]
    df['index'] = range(N)
    df['ticker'] = 'ticker'

    # Split the data equally for years 1992 to 2000
    num_years = 2000 - 1992 + 1
    rows_per_year = N // num_years
    df['date'] =  np.nan
    for year in range(1992, 2001):
        df.loc[rows_per_year * (year - 1992): rows_per_year * (year - 1992 + 1), 'date'] = pd.to_datetime(f'{year}-01-01')

    df['timestamp'] = df['date'].astype(str)

    for i in range(df.shape[0]):
        df.loc[df.index == i, 'vec_last'] = pd.Series([x[i, :]], index=[i])

    save_dest = 'logistic_sanity_check/tf/'
    os.makedirs(save_dest, exist_ok=True)
    save_dest_final = save_dest + 'data.tfrecord'
    with tf.io.TFRecordWriter(save_dest_final) as writer:
        for _, row in df.iterrows():
            example = serialize_news_link_all_single_news(row)
            writer.write(example)
    print('data created')
    return x, y, beta, save_dest, df['date']

if __name__ =='__main__':
    x,y,beta,save_dest,date = gen_data()


    ind_train = pd.to_datetime(date).dt.year <2000
    ind_test = pd.to_datetime(date).dt.year ==2000
    x_train = pd.DataFrame(x).loc[ind_train,:]
    y_train = np.sign(pd.DataFrame(y).loc[ind_train])
    x_test = pd.DataFrame(x).loc[ind_test,:]
    y_test = np.sign(pd.DataFrame(y).loc[ind_test])
    x_m = x_train.mean().values.reshape(1,-1)
    x_s = x_train.std().values.reshape(1,-1)
    x_train = (x_train-x_m)/x_s
    x_test = (x_test-x_m)/x_s


    m = LogisticRegression()
    m.fit(x_train,y_train)
    print('Simple model',m.score(x_test,y_test))


    par = Params()


    args = didi.parse()
    par = get_main_experiments(args.a, train_gpu=args.cpu == 0)



    start = time.time()

    par.grid.year_id = 2000
    par.train.sanity_check = True
    par.train.shrinkage_list = [0.0]
    par.train.norm = Normalisation.ZSCORE
    par.train.adam_rate = 0.05
    trainer = PipelineTrainer(par)
    trainer.def_create_the_datasets()
    if par.train.norm == Normalisation.ZSCORE:
        trainer.compute_parameters_for_normalisation()
    # train to find which penalisaiton to use
    # trainer.train_to_find_hyperparams()
    trainer.best_hyper = None
    trainer.train_on_val_and_train_with_best_hyper()
    end = time.time()
    print('Ran it all in ', np.round((end - start) / 60, 5), 'min', flush=True)
    df = trainer.get_prediction_on_test_sample()



    print('Simple model',m.score(x_test,y_test))