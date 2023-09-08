import nltk
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

if __name__ == "__main__":
    args = didi.parse()

    par = Params()
    data = Data(par)


    year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(data.p_eight_k_clean)]))
    print(year_list,flush=True)
    print(args.a,flush=True)
    year_todo = year_list[int(args.a)]

    type_txt = 'legal_' if args.legal ==0 else 'press_'
    f_name = f'{type_txt}{year_todo}.p'

    df = pd.read_pickle(data.p_eight_k_clean+f_name).reset_index(drop=True)
    if 'item' in df.columns:
        res = df[['cik','form_id','item']].copy()
        res['bow'] = np.nan
    else:
        res = df[['cik','form_id']].copy()
        res['bow'] = np.nan


    for i in tqdm.tqdm(df.index,f'Tokenise {type_txt}{year_todo}'):
        txt = df.loc[i,'txt']
        # Perform the cleaning steps
        bow = clean_from_txt_to_bow(txt)
        res.loc[i,'bow'] = json.dumps(bow)
        # code to get the dict again.
        # type(Counter(json.loads(json.dumps(bow))))
    res.to_pickle(data.p_eight_k_token+f_name)







