# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
import psutil
from didipack.trainer.trainer_ridge import TrainerRidge

from utils_local.trainer_specials import *

def set_ids_to_eight_k_df(df:pd.DataFrame,par:Params):
    df = df.reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
    df['id']=df['id'].astype(str)+'-'+df['news0'].astype(str)
    return df

def chose_trainer(par:Params):
    m = None
    if par.train.pred_model == PredModel.RIDGE:
        m = TrainerRidge(par)
    if par.train.pred_model == PredModel.LOGIT_EN:
        if (par.train.tnews_only is None) | (par.train.tnews_only == False):
            m = TrainerLogisticElasticNet(par, -1)
        else:
            m = TrainerLogisitcWithNewsInSample(par,para=-1)
    return m

def current_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # memory in GB

def normalize(x, par:Params):
    if par.train.norm == Normalisation.ZSCORE:
        x = (x - x.mean()) / x.std()
    if par.train.norm == Normalisation.RANK:
        x = x.rank(pct=True,axis=1)-0.5
    if par.train.norm == Normalisation.MINMAX:
        x = 2 * ((x - x.min()) / (x.max() - x.min())) - 1
    return x


def get_news_source_to_do_for_tfidf(par: Params):
    news_source_todo = [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD, NewsSource.WSJ_ONE_PER_STOCK] # default = all
    if par.tfidf.vocabulary_list == VocabularySetTfIdf.REUTERS_ONLY:
        news_source_todo = [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD] # what was done in old versions
    elif par.tfidf.vocabulary_list == VocabularySetTfIdf.WSJ_ONLY:
        news_source_todo = [NewsSource.EIGHT_PRESS,NewsSource.WSJ_ONE_PER_STOCK] # version wsj only
    return news_source_todo


def table_to_latex_complying_with_attila_totally_unreasonable_demands(df, rnd, paths, name):
    latex_str = df.round(rnd).to_latex(float_format=f"%.{rnd}f")

    lr = latex_str.split('tabular}{')[1].split('}')[0]
    latex_str = latex_str.replace(lr, lr[1:] + '}')
    latex_str = latex_str.replace(r'\begin{tabular}', r'\begin{tabular*}{1\linewidth}{@{\hskip\tabcolsep\extracolsep\fill}l*{1}')
    latex_str = latex_str.replace(r'\end{tabular}', r'\end{tabular*}')

    # Save the modified string to a file
    file_path = paths + name
    with open(file_path, 'w') as f:
        f.write(latex_str)
