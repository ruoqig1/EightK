import nltk
import pandas as pd
import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from collections import Counter
import sys
from parameters import *
from data import Data
import re
from utils_local.nlp_tokenize_and_bow import clean_from_txt_to_bow
import didipack as didi
import json
from utils_local.llm import EncodingModel
import pandarallel

if __name__ == "__main__":
    args = didi.parse()

    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    if socket.gethostname()=='3330L-214940-M':
        par.enc.opt_model_type = OptModelType.OPT_125m
        k = 2
    else:
        par.enc.opt_model_type = OptModelType.OPT_66b
        k = 2
    model = EncodingModel(par)

    par = Params()
    data = Data(par)
    print(args.a)
    print(args.thirdparty)
    print('--------',flush=True)
    # check if we are doing third party or not
    load_start = data.p_news_year if args.thirdparty == 0 else data.p_news_third_party_year
    save_start = data.p_vec_refinitiv if args.thirdparty == 0 else data.p_vec_third_party
    save_start = save_start+par.enc.opt_model_type.name+'/'

    year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(load_start)]))  #28 variations
    print('YEAR LIST', year_list)
    year_todo = year_list[args.a]


    print('START WORKING ON')
    print('Year',year_todo)
    print(f'Type of news, args.thirdparty={args.thirdparty}',flush=True)

    type_txt = 'ref_'  # if args.legal ==0 else 'press_'
    f_name = f'{type_txt}{year_todo}.p'
    df = pd.read_pickle(load_start+f_name).reset_index(drop=True)
    df['txt']=df['headline']+' \n '+df['body']
    df['date']=pd.to_datetime(df['timestamp'].apply(lambda x: x.split('T')[0]))
    df['m']=df['date'].dt.month
    unique_month = np.sort(np.unique(df['m']))
    df=df.set_index('id')[['m','txt']]
    df_grp = {name: group for name, group in df.groupby('m')}


    for month in unique_month:
        df = df_grp[month]
        res = df[['m']].copy()
        res['vec_last'] = np.nan
        res=res.drop(columns=['m'])
        # system to do in a few batch
        index_todo = np.array_split(df.index,int(np.ceil(df.shape[0]/k)))
        save_name = f'{type_txt}{year_todo}.p'
        for i in tqdm.tqdm(index_todo, f'Tokenise {save_name}'):
            txt_list = list(df.loc[i,'txt'].values)
            last_token_hidden_stage, _ = model.get_hidden_states_para(texts = txt_list)
            res.loc[i,'vec_last'] = pd.Series(last_token_hidden_stage,index=i)

        res.to_pickle(save_start+f_name)
        print('Saved in',save_start+f_name)