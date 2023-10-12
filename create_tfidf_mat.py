import pandas as pd
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from nltk.tokenize import word_tokenize
from parameters import *
from data import Data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import json
import joblib

def load_some_enc(par :Params):
    load_dir_and_file = par.get_vec_process_dir(merged_bow=True)
    df = pd.read_pickle(load_dir_and_file)
    return df


VV = [0.082527444, 0.116309986, 0.06505668, 0.18774417, 0.13331282, 0.45448, 0.08321442, 0.06508103, 0.08566995, 0.078396104, 1.0, 0.04918115, 0.07329284, 0.10165069, 0.12143111, 0.06417385, 0.16373745, 0.04719384, 0.111747965, 0.050367832, 0.0912352, 0.075461924, 0.11522688, 0.036300927, 0.065487206, 0.08774555, 0.07785033, 0.12217217, 0.074539706, 0.060614154, 0.09993455, 0.2253363, 0.08374144, 0.07242943, 0.096857, 0.06826295, 0.11101073, 0.09107542, 0.084456384, 0.111965254, 0.07930133, 0.14178361, 0.070804544, 0.11228882, 0.08428293, 0.12222331, 0.090765856, 0.12074542, 0.06775881, 0.09316684, 0.115149766, 0.11021345, 0.09925835, 0.07638243, 0.0504756, 0.085051656, 0.064806044, 0.098501354, 0.10192045, 0.08916644, 0.040259447, 0.10013352, 0.082170114, 0.089831844, 0.15904085, 0.10186933, 0.12016632, 0.12550822, 0.16149253, 0.11405565, 0.11323382, 0.092529714, 0.12694457, 0.17878851, 0.0758062, 0.084106356, 0.1166873, 0.06389326, 0.09377849, 0.13009627, 0.083022565, 0.09605385, 0.13116983, 0.12117517, 0.10490538, 0.08588973, 0.04951445, 0.13197848, 0.09688584, 0.0775485, 0.07258431, 0.1666584, 0.06402296, 0.00013315708, 0.06395128, 0.07150474, 0.074214384, 0.13231504, 0.09513728, 0.09332734, 0.045406073, 0.042482674, 0.11104748, 0.07797544, 0.10969914, 0.120893985, 0.10253927, 0.12527199, 0.11971544, 0.0447219, 0.07738249, 0.13973385, 0.11540498, 0.068244465, 0.09267378, 0.1143784, 0.05048898, 0.12443666, 0.13943286, 0.13653836, 0.102931246, 0.10109916, 0.13337827, 0.08426629, 0.061927676, 0.00020767128, 0.08375994, 0.1446245, 0.059553597, 0.18045619, 0.103632525, 0.07495906, 0.12131728, 0.100423425, 0.16013786, 0.100356966, 0.057219405, 0.08641984, 0.08567472, 0.10363418, 0.07034525, 0.067879565, 0.11666724, 0.07604714, 0.056174748, 0.07786216, 0.07381892, 0.069862664, 0.12974803, 0.13845275, 0.060154796, 0.048579384, 0.08959008, 0.21130042, 0.07921021, 0.074152365, 0.117589384, 0.10684629, 0.14040127, 0.106755525, 0.07004885, 0.07056762, 0.06740819, 0.08957253, 0.08615406, 0.07400063, 0.08898498, 0.04040072, 0.08937188, 0.0024256976, 0.085656166, 0.07210632, 0.052824397, 0.09100769, 0.13019681, 0.0, 0.09680514, 0.06760287, 0.07625836, 0.18013583, 0.056384325, 0.10526185, 0.11754743, 0.13965735, 0.070037395, 0.09824039, 0.07605913, 0.046932973, 0.11463441, 0.12308945, 0.08886692, 0.06892344, 0.06868823, 0.047087245, 0.07558915, 0.07332979, 0.04613489, 0.0988928, 0.061145134, 0.054480284]

if __name__ == '__main__':
    par = Params()
    data = Data(par)
    df = pd.DataFrame()
    par.enc.opt_model_type = OptModelType.BOW1
    all_text = []
    for news_source in [NewsSource.EIGHT_PRESS,NewsSource.NEWS_REF,NewsSource.NEWS_THIRD]:
        par.enc.news_source = news_source
        df = load_some_enc(par)
        all_text += df['vec_last'].values.tolist()

    all_text= all_text[:10000]

    decoded_bows = [json.loads(bow_str) for bow_str in tqdm.tqdm(all_text ,'decoding the bow')]

    # Convert bags of words into a term-document matrix
    print('Start building')
    dict_vectorizer = DictVectorizer()
    all_texts_matrix = dict_vectorizer.fit_transform(decoded_bows)
    print(all_texts_matrix.shape ,flush=True)
    # TF-IDF Transformation
    print('Start constructing the tfidf matrix')
    tfidf_transformer = TfidfTransformer(norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=False)
    tfidf_matrix = tfidf_transformer.fit_transform(all_texts_matrix)

    v = []
    for row_index_2 in range(200):
        similarity = cosine_similarity(tfidf_matrix[10,:], tfidf_matrix[row_index_2,:])
        v.append(similarity[0][0])

    v = np.array(v)
    VV = np.array(VV)

    np.corrcoef(v,VV)
    print(f"Cosine similarity between row 10 and row 20: {similarity[0][0]}")



