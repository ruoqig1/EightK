##########################################
#   Logistic Regression from SKLearn
##########################################

import didipack as didi
import joblib
import numpy as np
import pandas as pd
import os

from sklearnex import patch_sklearn

patch_sklearn()

import tqdm

from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates, get_chunks, get_tasks_for_current_node
import psutil
from utils_local.general import *
from utils_local.trainer_specials import *
from experiments_params import get_main_experiments
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed


class PipelineTrainer:
    def __init__(self, par: Params):
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.best_history = None
        self.par = par
        self.norm_params = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_dataset_with_id = None
        self.input_dim = None
        self.best_hyper = None
        self.model = None
        self.best_hyper_value = None

    def parse_tfrecord(self, example_proto):
        if self.par.enc.news_source in [NewsSource.NEWS_REF_ON_EIGHT_K]:
            feature_description = {
                'vec': tf.io.VarLenFeature(tf.float32),
                'id': tf.io.FixedLenFeature([], tf.string),
                'index': tf.io.FixedLenFeature([], tf.int64),
                'timestamp': tf.io.FixedLenFeature([], tf.string),
                'alert': tf.io.FixedLenFeature([], tf.int64),
                'ticker': tf.io.FixedLenFeature([], tf.string),
                'date': tf.io.FixedLenFeature([], tf.string),
                'permno': tf.io.FixedLenFeature([], tf.int64),
                'news0': tf.io.FixedLenFeature([], tf.int64),
                'ret': tf.io.FixedLenFeature([], tf.float32),
                'abret': tf.io.FixedLenFeature([], tf.float32),
                'cosine': tf.io.FixedLenFeature([], tf.float32),
                'm_cosine': tf.io.FixedLenFeature([], tf.float32),
                'prn': tf.io.FixedLenFeature([], tf.int64),
                'reuters': tf.io.FixedLenFeature([], tf.int64),
            }
        else:
            feature_description = {
                'vec': tf.io.VarLenFeature(tf.float32),
                'id': tf.io.FixedLenFeature([], tf.string),
                'index': tf.io.FixedLenFeature([], tf.int64),
                'timestamp': tf.io.FixedLenFeature([], tf.string),
                'alert': tf.io.FixedLenFeature([], tf.int64),
                'ticker': tf.io.FixedLenFeature([], tf.string),
                'date': tf.io.FixedLenFeature([], tf.string),
                'permno': tf.io.FixedLenFeature([], tf.int64),
                'ret': tf.io.FixedLenFeature([], tf.float32),
                'ret_m': tf.io.FixedLenFeature([], tf.float32),
                'reuters': tf.io.FixedLenFeature([], tf.int64),
            }

        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['vec'] = tf.sparse.to_dense(parsed_features['vec'])

        if self.norm_params is not None:
            if self.par.train.norm == Normalisation.ZSCORE:
                parsed_features['vec'] = (parsed_features['vec'] - self.norm_params['mean']) / tf.sqrt(
                    self.norm_params['var'] + 1e-7)
        return parsed_features

    @tf.autograph.experimental.do_not_convert
    def filter_start_year(self, x, const_start_year):
        return tf.greater_equal(tf.strings.to_number(tf.strings.substr(x['date'], 0, 4), out_type=tf.int32),
                                const_start_year)

    @tf.autograph.experimental.do_not_convert
    def filter_end_year(self, x, const_end_year):
        return tf.less_equal(tf.strings.to_number(tf.strings.substr(x['date'], 0, 4), out_type=tf.int32),
                             const_end_year)

    @tf.autograph.experimental.do_not_convert
    def filter_sample_based_on_par(self, x):
        reuters_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_reuters is None else tf.equal(
            x['reuters'], par.train.filter_on_reuters)
        prn_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_prn is None else tf.equal(x['prn'],
                                                                                                               par.train.filter_on_prn)
        alert_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_alert is None else tf.equal(
            x['alert'], par.train.filter_on_alert)
        cosine_condition = tf.constant(True, dtype=tf.bool) if self.par.train.filter_on_cosine is None else tf.greater(
            x['cosine'], x['m_cosine'] * (1 + tf.constant(par.train.filter_on_cosine, dtype=tf.float32)))
        combined_conditions = tf.logical_and(
            tf.logical_and(tf.logical_and(reuters_condition, prn_condition), alert_condition), cosine_condition)
        return combined_conditions

    @tf.autograph.experimental.do_not_convert
    def extract_input_and_label(self, x):
        abret = next(filter(x.__contains__, ('abret', 'ret_m')))
        return tf.reshape(x['vec'], (self.input_dim,)), tf.where(x[abret if self.par.train.abny else 'ret'] >= 0, 1, 0)

    @tf.autograph.experimental.do_not_convert
    def extract_with_id(self, x):
        abret = next(filter(x.__contains__, ('abret', 'ret_m')))
        return tf.reshape(x['vec'], (self.input_dim,)), tf.where(x[abret if self.par.train.abny else 'ret'] >= 0, 1, 0), \
            x['id'], x['date'], x['permno']

    @tf.autograph.experimental.do_not_convert
    def load_dataset(self, data_id, tfrecord_files, batch_size, start_year, end_year, return_id_too=False,
                     shuffle=False, batch=True):
        dataset = tf.data.TFRecordDataset(tfrecord_files)

        # Parse the dataset using the provided function
        dataset = dataset.map(self.parse_tfrecord)

        # Convert start_year and end_year to TensorFlow constants
        const_start_year = tf.constant(start_year, dtype=tf.int32)
        const_end_year = tf.constant(end_year, dtype=tf.int32)

        # Filter the dataset based on years
        dataset = dataset.filter(lambda x: self.filter_start_year(x, const_start_year))
        dataset = dataset.filter(lambda x: self.filter_end_year(x, const_end_year))

        # If this particular dataset must be filtered, we apply the filter at the tfrecords level.
        if self.par.train.apply_filter is not None:
            if data_id in self.par.train.apply_filter:
                dataset = dataset.filter(lambda x: self.filter_sample_based_on_par(x))

        if self.input_dim is None:
            # Take a sample to determine the input shape
            sample = next(iter(dataset.take(1)))
            self.input_dim = sample['vec'].shape[0]

        # Extract input x (vec_last) and label (sign of abret)
        if return_id_too:
            dataset = dataset.map(self.extract_with_id)
        else:
            dataset = dataset.map(self.extract_input_and_label)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)  # You can adjust the buffer size as needed

        # Batch the dataset
        if batch:
            dataset = dataset.batch(batch_size)
        return dataset

    def def_create_the_datasets(self, filter_func=lambda x: True, batch=True):

        path_with_records = self.par.get_training_dir()
        if (socket.gethostname() == '3330L-214940-M') & (self.par.train.sanity_check is None):
            path_with_records = path_with_records[:-1] + '_debug/'

        end_val = self.par.grid.year_id - 1
        start_val = end_val - self.par.train.T_val + 1
        start_train = self.par.grid.year_id - self.par.train.T_train
        end_train = start_val - 1
        start_test = self.par.grid.year_id
        end_test = self.par.grid.year_id - 1 + self.par.train.testing_window
        tfrecord_files = [os.path.join(path_with_records, f) for f in os.listdir(path_with_records) if '.tfrecord' in f]
        tfrecord_files = list(filter(filter_func, tfrecord_files))
        self.train_dataset = self.load_dataset('train', tfrecord_files, self.par.train.batch_size, start_train,
                                               end_train, shuffle=True, batch=batch)
        self.val_dataset = self.load_dataset('val', tfrecord_files, self.par.train.batch_size, start_val, end_val,
                                             batch=batch)
        self.test_dataset = self.load_dataset('test', tfrecord_files, self.par.train.batch_size, start_test, end_test,
                                              batch=batch)
        self.test_dataset_with_id = self.load_dataset('test', tfrecord_files, self.par.train.batch_size, start_test,
                                                      end_test, return_id_too=True, batch=batch)

        # for Logistic Regression
        print('Converting data to np', flush=True)
        self.X_train, self.y_train = extract_features_and_labels_to_np(trainer.train_dataset)
        self.X_test, self.y_test = extract_features_and_labels_to_np(trainer.test_dataset)
        self.X_val, self.y_val = extract_features_and_labels_to_np(trainer.val_dataset)

    def train_to_find_hyperparams(self):
        def evaluate_model(C, X_train, y_train, X_val, y_val):
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model',
                 LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=C, solver='saga', max_iter=65, verbose=1))
            ])
            pipe.fit(X_train, y_train)
            score = accuracy_score(y_val, pipe.predict(X_val))
            return C, score

        # Parallel grid search
        results = Parallel(n_jobs=2, pre_dispatch='1*n_jobs')(
            delayed(evaluate_model)(C, self.X_train, self.y_train, self.X_val, self.y_val) for C in
            self.par.train.shrinkage_list)

        best_coefficient, best_score = max(results, key=lambda x: x[1])
        self.best_hyper = best_coefficient

        print('Selected best penalisation', best_coefficient, flush=True)

    def train_on_val_and_train_with_best_hyper(self):
        print('Start the final training (train+val)', flush=True)
        # Train final model on combined training and validation sets with the best hyperparameter
        pipe_final = Pipeline([
            ('scaler', StandardScaler()),
            ('model',
             LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=self.best_hyper, solver='saga', max_iter=75,
                                n_jobs=1, verbose=1))
        ])

        # Combine the training and validation sets
        X_combined = np.concatenate([self.X_train, self.X_val])
        y_combined = np.concatenate([self.y_train, self.y_val])

        pipe_final.fit(X_combined, y_combined)
        self.model = pipe_final

    def get_prediction_on_test_sample(self):
        test_score = accuracy_score(self.y_test, self.model.predict(self.X_test))
        print(f"Test Accuracy: {test_score}")
        print(classification_report(self.y_test, self.model.predict(self.X_test)))
        print(confusion_matrix(self.y_test, self.model.predict(self.X_test)))

        final_model_pred = self.model.predict(self.X_test)  # Get the predictions
        final_model_pred_prb = self.model.predict_proba(self.X_test)

        # Get the probability estimates for each class
        ids = np.concatenate([id_batch.numpy().astype(str) for _, _, id_batch, _, _ in trainer.test_dataset_with_id],
                             axis=0)
        tickers = np.concatenate(
            [ticker_batch.numpy().astype(int) for _, _, _, _, ticker_batch in trainer.test_dataset_with_id], axis=0)
        dates = np.concatenate(
            [date_batch.numpy().astype(str) for _, _, _, date_batch, _ in trainer.test_dataset_with_id],
            axis=0)
        results_df = pd.DataFrame({
            'id': ids,
            'date': dates,
            'ticker': tickers,
            'y_true': self.y_test,
            'y_pred': final_model_pred,
            'y_pred_prb': final_model_pred_prb[:, 1]
        })
        results_df['accuracy'] = results_df['y_pred'] == results_df['y_true']

        return results_df


def parse_function(feature, label):
    return feature.numpy(), label.numpy()


def extract_features_and_labels_to_np(dataset):
    # Use parallel processing to map the parse function
    dataset = dataset.map(lambda x, y: tf.py_function(parse_function, [x, y], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Iterate over batches and concatenate results
    features, labels = zip(*list(dataset))
    return np.concatenate(features), np.concatenate(labels)


if __name__ == '__main__':
    # args = didi.parse()
    # print(args)
    # par = get_main_experiments(args.a, train_gpu=args.cpu == 0)
    for i in range(5, 8):
        par = get_main_experiments(i, train_gpu=True)
        par.enc.opt_model_type = OptModelType.OPT_125m
        par.enc.news_source = NewsSource.NEWS_SINGLE

        # Training args
        par.train.use_tf_models = True
        par.train.batch_size = 128

        start = time.time()

        # if socket.gethostname() == '3330L-214940-M':
        #     par.train.max_epoch = 1
        # else:
        #     par.train.max_epoch = 1
        #     par.train.T_train = 2
        #     par.train.T_val = 1
        #
        temp_save_dir = par.get_res_dir(s="logistic_regression")
        print('Model Directory ', temp_save_dir, flush=True)
        already_processed = os.listdir(temp_save_dir)
        save_name = f'{par.grid.year_id}.p'
        if save_name in already_processed:
            print(f'Already processed {save_name}', flush=True)
        else:
            print('Loading Datasets', flush=True)
            trainer = PipelineTrainer(par)
            trainer.def_create_the_datasets(
                filter_func=lambda x: 'mean' in x.split('/')[-1].split('_')
            )  # filter by 'mean' in file name

            print('Grid Search...', flush=True)
            trainer.train_to_find_hyperparams()
            trainer.train_on_val_and_train_with_best_hyper()

            # Evaluate the final model on the test set
            results_df = trainer.get_prediction_on_test_sample()

            end = time.time()

            # save the final model
            joblib.dump(trainer.model, temp_save_dir + save_name + '_model.joblib')
            print('Ran it all in ', np.round((end - start) / 60, 5), 'min', flush=True)

            results_df.to_pickle(temp_save_dir + save_name)
            par.save(temp_save_dir)

            print(results_df, flush=True)
            print('saved to', temp_save_dir + save_name, flush=True)
            print('We used', trainer.best_hyper_value)
