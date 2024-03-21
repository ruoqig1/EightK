#################################
#   Larger Model with 3 Layers
#################################


import didipack as didi
import numpy as np
import pandas as pd

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
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os


class PipelineTrainer:
    def __init__(self, par: Params):
        self.best_history = None
        self.par = par
        self.norm_params = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_dataset_with_id = None
        self.train_val_dataset = None
        self.input_dim = None
        self.best_hyper = None
        self.model = None
        self.best_hyper_value = None

        if self.par.train.tensorboard:
            self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch='1,3')
            print("tensorboard --logdir=" + os.path.abspath(self.log_dir))

    @staticmethod
    def compute_mean_variance(dataset, sample_rate=0.1):
        running_sum = 0
        running_squared_sum = 0
        num_samples = 0

        for batch, _ in tqdm.tqdm(dataset, 'Computing M-V on training sample'):
            # Randomly decide whether to include this batch based on sample rate
            # num_samples is slightly random, but this is to avoid iterating over the entire dataset for the
            # count of samples first. Given the large size of the dataset, this should be fine.
            if np.random.rand() < sample_rate:
                batch_size = batch.shape[0]
                running_sum += tf.reduce_sum(batch, axis=0)
                running_squared_sum += tf.reduce_sum(tf.square(batch), axis=0)
                num_samples += batch_size

        if num_samples == 0:
            raise ValueError("No samples were selected. Increase the dataset size or the sampling rate.")

        mean = running_sum / num_samples
        variance = (running_squared_sum / num_samples) - tf.square(mean)

        return mean.numpy(), variance.numpy()

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

    def filter_years(self, x, const_start_year, const_end_year):
        year = tf.strings.to_number(tf.strings.substr(x['date'], 0, 4), out_type=tf.int32)
        return tf.logical_and(tf.greater_equal(year, const_start_year),
                              tf.less_equal(year, const_end_year))

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
    def load_base_dataset(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord)

        if self.input_dim is None:
            # Take a sample to determine the input shape
            sample = next(iter(dataset.take(1)))
            self.input_dim = sample['vec'].shape[0]

        return dataset

    def extract_input_and_label(self, x):
        abret = next(filter(x.__contains__, ('abret', 'ret_m')))
        return tf.reshape(x['vec'], (self.input_dim,)), tf.where(x[abret if self.par.train.abny else 'ret'] >= 0, 1, 0)

    def extract_with_id(self, x):
        abret = next(filter(x.__contains__, ('abret', 'ret_m')))
        return tf.reshape(x['vec'], (self.input_dim,)), tf.where(x[abret if self.par.train.abny else 'ret'] >= 0, 1, 0), \
            x['id'], x['date'], x['permno']

    @tf.autograph.experimental.do_not_convert
    def load_dataset(self, data_id, dataset, batch_size, start_year, end_year, include_id=False,
                     shuffle=False, batch=True, cache=False):
        # Convert start_year and end_year to TensorFlow constants
        const_start_year = tf.constant(start_year, dtype=tf.int32)
        const_end_year = tf.constant(end_year, dtype=tf.int32)

        # Filter the dataset based on years
        dataset = dataset.filter(lambda x: self.filter_years(x, const_start_year, const_end_year))

        # If this particular dataset must be filtered, we apply the filter at the tfrecords level.
        if self.par.train.apply_filter is not None:
            if data_id in self.par.train.apply_filter:
                dataset = dataset.filter(lambda x: self.filter_sample_based_on_par(x))

        # Extract input x (vec_last) and label (sign of abret)
        dataset = dataset.map(self.extract_with_id if include_id else self.extract_input_and_label)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        if cache:
            dataset = dataset.cache()
        if batch:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        return dataset

    def train_model(self, tr_data, val_data, reg_to_use):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.par.train.patience,
            restore_best_weights=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.train.adam_rate)  # Using AMSGrad variant

        layers = [
            tf.keras.layers.Input(shape=(self.input_dim,))
        ]

        if self.par.train.norm == Normalisation.ZSCORE:
            # Create a Normalization layer
            normalization_layer = tf.keras.layers.Normalization(axis=-1)
            normalization_layer.adapt(tr_data.map(lambda x, y: x))
            layers.append(normalization_layer)

        layers.extend([
            tf.keras.layers.Dense(128, kernel_regularizer=reg_to_use),
            tf.keras.layers.LeakyReLU(alpha=0.01), 
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, kernel_regularizer=reg_to_use),
            tf.keras.layers.LeakyReLU(alpha=0.01),  
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, kernel_regularizer=reg_to_use),
            tf.keras.layers.LeakyReLU(alpha=0.01), 
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=reg_to_use)
        ])


        model = tf.keras.models.Sequential(layers)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()],
        )
        model.summary()

        callbacks = [early_stop]
        if self.par.train.tensorboard:
            callbacks.append(self.tensorboard_callback)

        # Train the model with early stopping
        history = model.fit(tr_data, validation_data=val_data, epochs=self.par.train.max_epoch, callbacks=callbacks)
        if val_data is not None:
            # Get the best validation AUC for this regularizer
            assert 'val_loss' in history.history.keys(), 'Validation set empty, problem with data splitting most likely.'
            min_val_loss = min(history.history['val_loss'])
            best_val_auc = max(history.history['val_auc'])
            return model, history
        else:
            return model

    def compute_parameters_for_normalisation(self):
        mean_vec_last, variance_vec_last = self.compute_mean_variance(self.train_dataset)
        self.norm_params = {'mean': mean_vec_last, 'var': variance_vec_last}
        print('Parameters for normalisation estimated on train sample', flush=True)

    def def_create_the_datasets(self, filter_func=lambda x: True, batch=True):

        path_with_records = self.par.get_training_dir()
        if (socket.gethostname() == '3330L-214940-M') & (self.par.train.sanity_check is None):
            path_with_records = path_with_records[:-1] + '_debug/'

        end_test = self.par.grid.year_id
        start_test = self.par.grid.year_id - 1 + self.par.train.testing_window
        end_val = start_test - 1
        start_val = end_val - self.par.train.T_val + 1
        end_train = start_val - 1
        start_train = end_train - self.par.train.T_train + 1
        tfrecord_files = [os.path.join(path_with_records, f) for f in os.listdir(path_with_records) if '.tfrecord' in f]
        tfrecord_files = list(
            filter(lambda x: start_train <= int(x.split('/')[-1].split('_')[0]) <= end_test, tfrecord_files)
        )
        tfrecord_files = list(filter(filter_func, tfrecord_files))
        base_dataset = self.load_base_dataset(tfrecord_files)

        self.train_dataset = self.load_dataset('train', base_dataset, self.par.train.batch_size, start_train,
                                               end_train, shuffle=True, batch=batch, cache=True)
        self.val_dataset = self.load_dataset('val', base_dataset, self.par.train.batch_size, start_val, end_val,
                                             batch=batch)
        self.test_dataset = self.load_dataset('test', base_dataset, self.par.train.batch_size, start_test, end_test,
                                              batch=batch)
        self.test_dataset_with_id = self.load_dataset('test', base_dataset, self.par.train.batch_size, start_test,
                                                      end_test, include_id=True, batch=batch)
        self.train_val_dataset = self.load_dataset('train_val', base_dataset, self.par.train.batch_size, start_train,
                                                      end_val, shuffle=True, batch=batch, cache=True)

    def train_to_find_hyperparams(self):
        best_reg = None
        best_history = None
        metric_direction = -1 if self.par.train.monitor_metric == 'loss' else 1
        best_metric = -float('inf') * metric_direction
        best_reg_value = None
        for reg_value in self.par.train.shrinkage_list:
            if self.par.train.l1_ratio[0] == 1.0:
                reg = tf.keras.regularizers.l2(reg_value)
            if self.par.train.l1_ratio[0] == 0.0:
                reg = tf.keras.regularizers.l1(reg_value)
            if self.par.train.l1_ratio[0] == 0.5:
                reg = tf.keras.regularizers.l1_l2(reg_value, reg_value)
            model, history = self.train_model(tr_data=self.train_dataset, val_data=self.val_dataset,
                                              reg_to_use=reg)
            print('Evaluate the model on the val_dataset', flush=True)
            model.evaluate(self.val_dataset)
            print('Evaluate the model on the train_dataset', flush=True)
            model.evaluate(self.train_dataset)

            model_metric = max(metric_direction * np.array(history.history[self.par.train.monitor_metric]))
            if model_metric > best_metric:
                best_metric = model_metric
                best_reg = reg
                best_history = history
                best_reg_value = reg_value
        self.best_hyper = best_reg
        self.best_hyper_value = best_reg_value
        self.best_history = best_history

        print('Selected best penalisation', best_reg_value, flush=True)

    def train_on_val_and_train_with_best_hyper(self):
        print('Start the final training (train+val)', flush=True)
        # Train the model using the best hyperparameters found
        self.model = self.train_model(tr_data=self.train_val_dataset, val_data=None, reg_to_use=self.best_hyper)

    def get_prediction_on_test_sample(self):
        @tf.autograph.experimental.do_not_convert
        def extract_features_for_prediction(x, y, ids, dates, tickers):
            return x

        @tf.autograph.experimental.do_not_convert
        def extract_features_and_labels(x, y, ids, dates, tickers):
            return x, y

        # Predict on the test dataset
        predictions = self.model.predict(self.test_dataset_with_id.map(extract_features_for_prediction))

        # Extract true labels, IDs, tickers, and dates in one pass
        true_labels, ids, tickers, dates = [], [], [], []
        for _, y_batch, id_batch, date_batch, ticker_batch in self.test_dataset_with_id:
            true_labels.append(y_batch.numpy())
            ids.append(id_batch.numpy().astype(str))
            tickers.append(ticker_batch.numpy().astype(int))
            dates.append(date_batch.numpy().astype(str))

        true_labels = np.concatenate(true_labels, axis=0)
        ids = np.concatenate(ids, axis=0)
        tickers = np.concatenate(tickers, axis=0)
        dates = np.concatenate(dates, axis=0)

        # Compute the predicted labels
        predicted_labels = (predictions > 0.5).astype(int).flatten()

        # Saving the results with corresponding IDs, dates, and tickers
        df = pd.DataFrame({
            'id': ids,
            'date': dates,
            'ticker': tickers,
            'y_true': true_labels,
            'y_pred': predicted_labels,
            'y_pred_prb': predictions.flatten()
        })
        df['accuracy'] = df['y_pred'] == df['y_true']

        # evaluation_results = self.model.evaluate(
        #     self.test_dataset,
        #     return_dict=True
        # )
        # print('####### SANITY CHECK')
        # if 'accuracy' in evaluation_results:
        #     print("Accuracy from .evaluate:", evaluation_results['accuracy'], flush=True)
        # if 'auc' in evaluation_results:
        #     print("AUC from .evaluate:", evaluation_results['auc'], flush=True)
        print("Accuracy from .df:", df['accuracy'].mean().round(6), flush=True)

        return df


if __name__ == '__main__':
    # args = didi.parse()
    # print(args)
    # par = get_main_experiments(args.a, train_gpu=args.cpu == 0)
    for i in range(7):
        par = get_main_experiments(i, train_gpu=True)
        par.enc.opt_model_type = OptModelType.OPT_125m
        par.enc.news_source = NewsSource.NEWS_SINGLE

        # Training args
        par.train.use_tf_models = True
        par.train.l1_ratio = [0.5]
        par.train.batch_size = 32
        par.train.monitor_metric = 'val_auc'
        par.train.patience = 5
        par.train.max_epoch = 12
        par.train.adam_rate = 0.0001

        par.train.tensorboard = True

        # par.train.T_train = 1  # reduce the dataset size for faster training
        
        # Skip Normalisation
        # par.train.norm = None

        start = time.time()

        # if socket.gethostname() == '3330L-214940-M':
        #     par.train.max_epoch = 1
        # else:
        #     par.train.max_epoch = 1
        #     par.train.T_train = 2
        #     par.train.T_val = 1
        #
        temp_save_dir = par.get_res_dir()
        print('Model Directory ', temp_save_dir, flush=True)
        already_processed = os.listdir(temp_save_dir)
        save_name = f'{par.grid.year_id}.p'
        if save_name in already_processed and 0:
            print(f'Already processed {save_name}', flush=True)
        else:

            trainer = PipelineTrainer(par)
            trainer.def_create_the_datasets(
                filter_func=lambda x: 'mean' not in x.split('/')[-1].split('_'),  # filter by 'mean' in file name
            )
            print('Preprocessing time', np.round((time.time() - start) / 60, 5), 'min', flush=True)
            # train to find which penalisation to use
            # trainer.train_to_find_hyperparams()
            # trainer.train_on_val_and_train_with_best_hyper()

            # This is a known good regulariser to save from testing for it everytime.
            trainer.model = trainer.train_model(tr_data=trainer.train_val_dataset, val_data=None, reg_to_use=tf.keras.regularizers.l1_l2(0.000005, 0.000005))
            
            end = time.time()
            trainer.model.save(temp_save_dir + save_name + '_model.keras')
            print('Ran it all in ', np.round((end - start) / 60, 5), 'min', flush=True)
            df = trainer.get_prediction_on_test_sample()
            df.to_pickle(temp_save_dir + save_name)
            par.save(temp_save_dir)

            print(df, flush=True)
            print('saved to', temp_save_dir + save_name, flush=True)
            print('We used', trainer.best_hyper_value)
