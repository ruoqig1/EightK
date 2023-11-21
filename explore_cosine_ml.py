import didipack.utils_didi.parser
import pandas as pd

from vec_main import load_and_process_eight_k_legal_or_pressed
import tqdm
from didipack import PandasPlus
from statsmodels import api as sm
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from didipack import OlsPLus
from sklearn.metrics import roc_curve, auc, roc_auc_score
from parameters import BRYAN_MAIN_CATEGORIES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from experiments_params import get_experiments_coverage_pred
import didipack as didi

import shap
from explore_cosine_correlations import load_coverage_as_in_first_paper
from parameters import BRYAN_MAIN_CATEGORIES

def add_bryan_predictors(res,predictors_col, par:Params):
    if par.covpred.use_age_and_market_only is not None:
        predictors_col = predictors_col +['age','market_equity']
        res = res+ [['age','age'],['market_equity','market_equity']]
    else:
        for k in BRYAN_MAIN_CATEGORIES.keys():
            predictors_col = predictors_col + BRYAN_MAIN_CATEGORIES[k]
            for c in BRYAN_MAIN_CATEGORIES[k]:
                res.append([k, c])
    return res,predictors_col


def get_predictors_list(par:Params):
    cont_cov = ['cov_sum_l', 'cov_pct_l']
    if par.covpred.contemp_cov is not None:
        cont_cov = ['cov_sum', 'cov_pct']
    if par.covpred.predictors == PredictorsCoverage.ALL:
        predictors_col = cont_cov
        res = [['Coverage', cont_cov[0]], ['Coverage', cont_cov[1]]]
        res,predictors_col = add_bryan_predictors(res,predictors_col,par)
    if par.covpred.predictors == PredictorsCoverage.COVE_ONLY:
        predictors_col = cont_cov
        res = [['Coverage', cont_cov[0]], ['Coverage', cont_cov[1]]]
    if par.covpred.predictors == PredictorsCoverage.ALL_BUT_COV:
        predictors_col = []
        res = []
        res,predictors_col = add_bryan_predictors(res,predictors_col,par)
    if par.covpred.predictors in [PredictorsCoverage.ITEMS_NAMES,PredictorsCoverage.ITEMS_NAMES_AND_SIZE]:
        item_list = [1.01, 1.02, 1.04, 2.01, 2.03, 2.04, 2.05, 2.06, 3.02, 3.03, 4.01, 4.02, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08, 6.02, 6.03, 6.04, 6.05, 7.01, 8.01]
        predictors_col = item_list
        res = [['items',x] for x in item_list]
        if par.covpred.predictors == PredictorsCoverage.ITEMS_NAMES_AND_SIZE:
            predictors_col = predictors_col + ['market_equity']
            res =res + [['size', 'market_equity']]
    res = pd.DataFrame(res, columns=['Category', 'Name'])
    return res,predictors_col

def normalize_df(df):
    if par.covpred.normalize == Normalisation.ZSCORE:
        for c in tqdm.tqdm(predictors_col,'normalizing'):
            df[c] = (df[c]-df[c].mean())/df[c].std()
            df[c] = df[c].fillna(0.0)
    if par.covpred.normalize == Normalisation.RANK:
        for c in tqdm.tqdm(predictors_col,'normalizing'):
            df[c]= df.groupby('form_date')[c].rank(pct=True)-0.5
            df[c] = df[c].fillna(0.0)
    return df

def plot_roc(fpr,tpr):
    # Compute ROC curve and AUC
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()


def get_shapeley_values_individual(base_accuracy,predictors_col,clf,X_test):
    nb_shap = 20
    shap = {}
    for c in tqdm.tqdm(predictors_col,'home made shap'):
        acc = []
        for _ in range(nb_shap):
            x_shap = X_test.copy()
            np.random.shuffle(x_shap[c].values)
            shap_pred = clf.predict(x_shap.values)
            acc.append(base_accuracy- accuracy_score(y_test['covered'], shap_pred))
        shap[c] = acc
    shap = pd.Series(shap)
    m= shap.apply(np.mean)
    q =.25
    ql = shap.apply(lambda x: np.quantile(x,q))
    qh = shap.apply(lambda x: np.quantile(x,1-q))
    sh = pd.DataFrame({'m':m,'ql':ql,'qh':qh})
    sh=sh.sort_values('m',ascending=True)
    # sh = sh.loc[(sh['ql']>=0) & (sh['m']>0),:]
    sh['error'] = sh['qh'] - sh['ql']
    sh*=100
    return sh

def get_shapeley_values_categories(base_accuracy,res,clf,X_test):
    cat_dict = res.groupby('Category')['Name'].unique()
    nb_shap = 20
    shap = {}
    for c in tqdm.tqdm(cat_dict.keys(),'home made shap by cat'):
        acc = []
        for _ in range(nb_shap):
            x_shap = X_test.copy()
            for col in cat_dict[c]:
                np.random.shuffle(x_shap[col].values)
            shap_pred = clf.predict(x_shap.values)
            acc.append(base_accuracy- accuracy_score(y_test['covered'], shap_pred))
        shap[c] = acc
    shap = pd.Series(shap)
    m= shap.apply(np.mean)
    q =.25
    ql = shap.apply(lambda x: np.quantile(x,q))
    qh = shap.apply(lambda x: np.quantile(x,1-q))
    sh = pd.DataFrame({'m':m,'ql':ql,'qh':qh})
    sh=sh.sort_values('m',ascending=True)
    # sh = sh.loc[(sh['ql']>=0) & (sh['m']>0),:]
    sh['error'] = sh['qh'] - sh['ql']
    sh*=100
    return sh

def plot_home_made_shapeley(sh, base_accuracy,dumb_accuracy):
    # Plot
    plt.figure(figsize=(6, 10))
    plt.errorbar(sh['m'], sh.index, xerr=sh['error'] / 2, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xlabel('Values')
    plt.ylabel('Contribution To Accuracy (in %)')

    plt.title(f'Model Gain Over Naive: {np.round((base_accuracy-dumb_accuracy)*100,2)}%')
    plt.grid(True)
    plt.tight_layout()

if __name__ == '__main__':
    args = didi.parse()
    np.random.seed(12345)
    par = get_experiments_coverage_pred(args.a)


    par.get_coverage_predict_save_dir()
    # par.covpred.small_sample = True
    data = Data(par)
    res, predictors_col = get_predictors_list(par)
    df = load_coverage_as_in_first_paper(data)
    if par.covpred.small_sample:
        df = df.reset_index(drop=True)
        keep = np.random.choice(df.index,100000,replace=False)
        df = df.loc[keep,:]
        print('run on subsample', df.shape)
    else:
        df = df.reset_index(drop=True)

    df= df[['covered']+predictors_col+['form_date']]
    df = normalize_df(df)
    save_dir = par.get_coverage_predict_save_dir()
    # split sample
    df = df.reset_index(drop=True)
    X = df[predictors_col]
    y = df['covered']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    test = np.random.choice(df.index, int(np.round(df.shape[0]*.2)), replace=False)
    X_test = X.loc[test,:]
    y_test = y.loc[test]
    X_train = X.loc[~X.index.isin(test),:]
    y_train = y.loc[~y.index.isin(test)]

    # Initialize and train the model
    clf = RandomForestClassifier(random_state=42,n_jobs=-1)

    print('Start training on', X_train.shape[0]/1e6,'mil training sample and',X_train.shape[1], 'features')
    clf.fit(X_train.values, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test.values)
    y_test = pd.DataFrame(y_test)
    y_pred_proba = clf.predict_proba(X_test.values)[:, 1]
    y_test['pred']=y_pred
    y_test['pred_prb']=y_pred_proba

    base_accuracy = accuracy_score(y_test['covered'], y_pred)
    dumb_accuracy = accuracy_score(y_test['covered'], np.ones_like(y_pred))

    # process ytest for saving

    fpr, tpr, _ = roc_curve(y_test['covered'], y_pred_proba)
    plot_roc(fpr,tpr)
    plt.savefig(save_dir+'roc.png')
    plt.close()





    # my own custom shap
    sh_cat = get_shapeley_values_categories(base_accuracy,res,clf,X_test)
    plot_home_made_shapeley(sh_cat, base_accuracy, dumb_accuracy)
    plt.savefig(save_dir+'shap_cat.png')
    plt.close()


    # sh_ind = get_shapeley_values_individual(base_accuracy,predictors_col,clf,X_test)
    # plot_home_made_shapeley(sh_ind, base_accuracy, dumb_accuracy)
    # plt.savefig(save_dir+'shap_ind.png')
    # plt.close()

    sh_cat.to_pickle(save_dir+'sh_cat.p')
    # sh_ind.to_pickle(save_dir+'sh_ind.p')
    res.to_pickle(save_dir+'res.p')
    y_test.to_pickle(save_dir+'y_test.p')

