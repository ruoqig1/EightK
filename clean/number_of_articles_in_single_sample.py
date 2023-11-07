import pandas as pd
import tensorflow as tf
import os

import tqdm

from parameters import *
from data import Data

par = Params()
data = Data(par)

dir = load_dir = data.p_to_vec_main_dir + '/single_stock_news_to_vec/'

third = pd.DataFrame()
ref = pd.DataFrame()
for f in tqdm.tqdm(os.listdir(dir)):
    df = pd.read_pickle(dir+f)
    r = pd.Series({'alert':df['alert'].sum(),'other':(df['alert']==False).sum()},name=f)
    if 'ref' in f:
        ref = pd.concat([ref,r],axis=1)
    else:
        third = pd.concat([third,r],axis=1)

ref.sum(1)/1e6
third.sum(1)/1e6
(ref.sum(1)+third.sum(1))/1e6
third.sum()

ref = ref.T
third = third.T
ref = ref.reset_index()
third = third.reset_index()
ref['index'] = ref['index'].apply(lambda x: int(x.split('ref')[1].split('.p')[0]))
third['index'] = third['index'].apply(lambda x: int(x.split('third')[1].split('.p')[0]))

a= ref.loc[ref['index']<=2019,:].sum()/1e6
b= third.loc[third['index']<=2019,:].sum()/1e6