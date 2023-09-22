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
import pandarallel

if __name__ == "__main__":
    args = didi.parse()


    par = Params()
    data = Data(par)
    print(args.a)
    print(args.thirdparty)
    print('--------',flush=True)
    # check if we are doing third party or not
    load_start = data.p_news_year if args.thirdparty == 0 else data.p_news_third_party_year
    save_start = data.p_news_token_year if args.thirdparty == 0 else data.p_news_third_party_token_year

    year_list = np.unique(np.sort([int(f.split('_')[1].split('.')[0]) for f in os.listdir(load_start)]))  #28 variations
    print('YEAR LIST', year_list)
    print(args.a,type(args.a))
    year_todo = year_list[args.a]
    print('here')

    type_txt = 'ref_'# if args.legal ==0 else 'press_'
    f_name = f'{type_txt}{year_todo}.p'

    print('START WORKING ON')
    print('Year',year_todo)
    print(f'Type of news, args.thirdparty={args.thirdparty}',flush=True)



    df = pd.read_pickle(load_start+f_name).reset_index(drop=True)
    res = df[['id']].copy()
    txt_type = ['body','headline']
    for c in txt_type:
        res[c]=np.nan


    for i in tqdm.tqdm(df.index,f'Tokenise {type_txt}{year_todo}'):
        for c in txt_type:
            txt = df.loc[i,c]
            # Perform the cleaning steps
            bow = clean_from_txt_to_bow(txt)
            res.loc[i,c] = json.dumps(bow)
            if i in [0,10,100]:
                print(i,bow,flush=True)
                print('#'*50)
        # code to get the dict again.
        # type(Counter(json.loads(json.dumps(bow))))
    res.to_pickle(save_start+f_name)







