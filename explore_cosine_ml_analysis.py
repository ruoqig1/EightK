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

    # Plotting the ROC curve
    plt.figure()

    pred = []
    for i in range(4):
        par = get_experiments_coverage_pred(i)
        load_dir = par.get_coverage_predict_save_dir()
        os.listdir(load_dir)
        y_test = pd.read_pickle(load_dir+'y_test.p')

        fpr, tpr, _ = roc_curve(y_test['covered'], y_test['pred_prb'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{par.covpred.predictors.name} (area = %0.2f)' % roc_auc)
        r = y_test['pred_prb']
        r.name = par.covpred.predictors.name
        pred.append(r)
    c = pd.concat(pred,axis=1)
    print(c.corr())
    y_test['pred_prb'] = c.drop(columns='ALL').mean(1)
    fpr, tpr, _ = roc_curve(y_test['covered'], y_test['pred_prb'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Ensembling (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()