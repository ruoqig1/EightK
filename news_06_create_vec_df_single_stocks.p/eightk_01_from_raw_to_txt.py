import pandas as pd
import tqdm
import sys
from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, PlotPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pyperclip
from cleaning import load_csv
from bs4 import BeautifulSoup
import re


def get_description(txt):
    if '<DESCRIPTION>' in txt:
        desc=file_text.split('<DESCRIPTION>')[1].split('<')[0]
        k=1
    else:
        desc = np.nan
        k=0
    return desc, k

def found_number_of_section_in_some_string(string_with_items:str):
    pattern = r'\b(' + '|'.join([str(x) for x in Constant.LIST_ITEMS_FULL]) + r')\b'
    search_result = re.search(pattern, string_with_items)
    if search_result:
        return float(search_result.group(1))
    else:
        return np.nan


def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)






def get_items_number_and_text_clean(list_of_item_locations_string:[], idx: int):
    item = (found_number_of_section_in_some_string(list_of_item_locations_string[idx]))
    if idx == len(list_of_item_locations_string) - 1:
        txt = file_text.split(list_of_item_locations_string[idx])[-1]
    else:
        txt = file_text.split(list_of_item_locations_string[idx])[-1]
        txt = txt.split(list_of_item_locations_string[idx + 1])[0]
    txt = list_of_item_locations_string[idx] + txt
    txt = try_to_parse_html(txt)
    return txt, item

def find_item_location_variants(text, x=50):
    numbers = [str(num) for num in Constant.LIST_ITEMS_FULL]

    pattern = f'>\\s*(&#\\d+;)?\\s*(Item|ITEM)(?P<after>[\\s\\S]{{0,{x}}})'
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        pattern = f'(\\n|\\r\\n)\\s*(Item|ITEM)(?P<after>[\\s\\S]{{0,{x}}})'
        matches = re.findall(pattern, text)

    if len(matches) == 0:
        numbers_pattern = '|'.join(numbers)
        pattern = f'(>|\\n|\\r\\n)\\s*({numbers_pattern})(?P<after>[\\s\\S]{{0,{x}}})'
        matches = re.findall(pattern, text)

    # conactening it all into one big string
    extended_matches = [a + b + c for a, b, c in matches]
    return extended_matches

def process_main_eight_k_legal_file(file_text,cik,link,form_id):
    try:
        # we find all the place where an item is mentioned
        item_locations = find_item_location_variants(file_text)
        # for each of those items, we...
        res = []
        for i in range(len(item_locations)):
            # extract the item definition and the clean text going with it
            # list_of_item_locations_string = item_locations; idx = i
            txt, item = get_items_number_and_text_clean(list_of_item_locations_string=item_locations, idx=i)
            res.append(pd.Series({
                'cik':cik,
                'form_id':form_id,
                'link':link,
                'item': item,
                'txt': txt,
                'ran': True
            }))
        assert len(item_locations)>0, 'pass to the except as we couldnt find enough items locations'
    except:
        # if we can't process it we return the id, main text, and everything else with a bool to mark it as fail. In main text, we put the full file txt to simplify debugging
        res =[pd.Series({
            'cik': cik,
            'form_id': form_id,
            'link': link,
            'item': np.nan,
            'txt': file_text,
            'ran': False
        })]
    try:
        res = pd.concat(res, axis=1).T
    except:
        breakpoint()
    return res


def check_for_99_1_variations_in_document_tags(text):
    sp = text.split('<TEXT>')
    is_press_release = False
    is_pdf = False
    if len(sp)>1:
        # Regular expression to match variations of 99.1, including 991, 99_1, 99-1, etc.
        pattern = r'99[._\-]?1'
        # Use re.search to check if the pattern exists in the text
        is_press_release = re.search(pattern, sp[0]) is not None

        # Use re.search to check if the pattern exists in the text
        is_pdf = '.pdf' in sp[0].lower()
    return is_press_release, is_pdf


def check_for_press_release_in_document_tags(text):
    sp = text.split('<TEXT>')
    is_press_release = False
    if len(sp) > 1:
        # Regular expression to match variations of 99.1, including 991, 99_1, 99-1, etc.
        number_pattern = r'99[._\-]?1'

        # Keywords that might indicate a press release in the title or description
        keyword_pattern = r'(press\s*release|announcement|news\s*update|public\s*statement|official\s*release)'

        # Combined pattern to search for both number variations and keywords
        combined_pattern = fr'({number_pattern}|{keyword_pattern})'

        # Use re.search to check if any of the patterns exist in the text before the <TEXT> tag
        is_press_release = re.search(combined_pattern, sp[0], re.IGNORECASE) is not None
    return is_press_release


def is_readable(text):
    # Count all alphabetic and standard punctuation characters
    readable_chars = len(re.findall('[a-zA-Z0-9\s,;.!?\-\'"]+', text))
    total_chars = len(text)

    # Check for a minimum number of total characters to avoid false positives on very short texts
    if total_chars < 50:
        return False

    # Calculate the percentage of readable characters in the text
    readable_percentage = (readable_chars / total_chars) * 100

    # If the majority of characters are readable, consider the text readable
    return readable_percentage > 50

from bs4 import BeautifulSoup, NavigableString
def try_to_parse_html(txt):
    try:
        txt_parsed = BeautifulSoup(txt, "html.parser").getText()
    except:
        txt_parsed = np.nan
    return txt_parsed

def extract_text_if_press_relase(file_text,k,cik,form_id,link):
    try:
        # for the follow up documents, we try to detect if it's a press release.
        is_press, is_pdf = check_for_99_1_variations_in_document_tags(file_text)
        if is_press:
            parsed_text = try_to_parse_html(file_text)
            res_vec= pd.Series({
                'cik': cik,
                'form_id': form_id,
                'link': link,
                'k': k,
                'pdf': is_pdf,
                'txt': parsed_text,
                'readable': is_readable(parsed_text),
                'ran':True
            })
        else:
            res_vec = np.nan
    except:
        is_press = True
        res_vec = pd.Series({
            'cik': cik,
            'form_id': form_id,
            'link': link,
            'k': k,
            'pdf': np.nan,
            'txt': file_text, # give raw txt so we can debug easily
            'readable': np.nan,
            'ran':False
        })
    return is_press, res_vec

if __name__ == "__main__":

    try:
        grid_id = int(sys.argv[1])
        print('Running with args',grid_id,flush=True)
    except:
        print('Debug mode on local machine')
        grid_id = -1

    # launch 20 different ones.
    year_todo =np.arange(2004,2024,1)[grid_id]

    par = Params()
    data = Data(par)


    # icf = load_csv()
    if Constant.IS_VM:
        csv = pd.read_csv('/mnt/layline/datasets/currentReports/currentReports.csv')
        load_dir = '/mnt/layline/edgar/forms/8-K/'
        save_dir = f'res/8k_clean/'
        print('RUNNING IN VM MODE',flush=True)
    else:
        csv = pd.read_csv('/Users/adidisheim/Dropbox/AB-AD_Share/current2023/currentReports.csv')
        load_dir = '/Users/adidisheim/Dropbox/AB-AD_Share/current2023/8-K/'
        save_dir = Constant.DROP_RES_DIR + f'/8k_clean/'
    os.makedirs(save_dir,exist_ok=True)


    csv['acceptanceDatetime'] = pd.to_datetime(csv['acceptanceDatetime'].astype(str).str[:-2], format='%Y%m%d%H%M%S', errors='coerce')
    csv['atime'] = csv['acceptanceDatetime'].dt.time
    csv['adate'] = pd.to_datetime(csv['acceptanceDatetime'].dt.date)
    csv = csv.dropna(subset='adate')
    csv['accessionNumber'] = csv['accessionNumber'].apply(lambda x: str(x.replace('-', '')))

    icf = csv.loc[csv['adate'].dt.year==year_todo,:]

    print(f'Starting on year {year_todo}')
    print(f'Total todo',icf.shape)

    icf = icf.dropna(subset=['sic','accessionNumber'])


    legal_df = pd.DataFrame()
    press_df = pd.DataFrame()
    press_df_bug = pd.DataFrame()
    # loopingf through the icf document with source
    for l in tqdm.tqdm(icf.index):
    # for l in tqdm.tqdm(icf.index[:100]):
        # select cik and form number as it is the way the folder is organised
        cik = str(int(icf.loc[l,'cik']))
        form_id = str(icf.loc[l,'accessionNumber'].replace('-',''))
        # path to the data
        f = load_dir+ f"{cik}/{form_id}/form.txt"
        # get the link
        link =icf.loc[l,'URL']
        # print(l, link)
        # check that indeed the file does exist and we do have a match
        if os.path.exists(f):
            form = open(f, "r").read()
            # the 8k is organised in documents, so we split across file name to have individual documents use to do it with FILENAME
            for k, file_text in enumerate(form.split('<DOCUMENT>')):
                # k==1 means we are in the document supposed to contain all the 8k info (items and brief legal descriptio nof each items
                press_count = 0
                already_one_press_df =False
                if k ==1:
                    legal_int = process_main_eight_k_legal_file(file_text=file_text,cik=cik,link=link,form_id=form_id)
                    legal_df = pd.concat([legal_df,legal_int],axis=0)
                elif k in [2,3]:
                    is_press, press_txt_vec = extract_text_if_press_relase(file_text, k, cik, form_id, link)
                    if is_press:
                        press_df = pd.concat([press_df,press_txt_vec],axis=1)

    press_df = press_df.T.reset_index(drop=True)
    legal_df = legal_df.reset_index(drop=True)

    press_df.to_pickle(save_dir+f'press_{year_todo}.p')
    legal_df.to_pickle(save_dir+f'legal_{year_todo}.p')

    print(f'Finish saving press_df of shape {press_df.shape}, and legal_df {legal_df.shape}')

 # username@vm-172-26-151-140.desktop.cloud.unimelb.edu.au
 # ssh-copy-id ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au
# https://vm-172-26-151-140.desktop.cloud.unimelb.edu.au:3300/client/connect;id=73402b077a2a70806d094026329d227a