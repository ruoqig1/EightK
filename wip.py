import pandas as pd
import tensorflow as tf
import os
from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm

from parameters import *
from data import Data
from matplotlib import pyplot as plt
par = Params()
data = Data(par)
press_or_legal = 'legal'
df =pd.read_pickle(data.p_dir+f'ss_count_{press_or_legal}.p')
press =pd.read_pickle(data.p_dir+f'ss_count_press.p')


ind = df['items'].isin(Constant.LIST_ITEMS_TO_USE)
df = df.loc[ind,:]

ind = press['items'].isin(Constant.LIST_ITEMS_TO_USE)
press = press.loc[ind,:]
df.groupby('year')['len'].median().plot()
plt.show()
press.groupby('year')['len'].median().plot()
plt.show()

# df['len'].quantile(np.arange(0.01,1,0.01)).plot()
# plt.show()
#
# PandasPlus.winzorize_series(press['len'],2).quantile(np.arange(0.01,1,0.01)).plot()
# plt.show()

l = df.groupby(['year','items'])['len'].median().reset_index().pivot(columns='items',values='len',index='year')
l_press = press.groupby(['year','items'])['len'].median().reset_index().pivot(columns='items',values='len',index='year')
nb = df.groupby(['year','items'])['len'].count().reset_index().pivot(columns='items',values='len',index='year')

col = nb.mean()
col = [x for x in col.index if col[x]>1000]
l[col].plot()
plt.show()

l_press[col].plot()
plt.show()


nb_perc = nb/nb.iloc[8,:].values.reshape(1,-1)
nb_perc[col].plot()
plt.show()

nb[col].plot()
plt.show()

