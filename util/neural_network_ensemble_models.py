import re
import sys
import nltk
import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import initializers

from .preprocess_log_regr_and_svm import generate_features_train_data, generate_features_test_data
from .preprocess_neural_network_ensemble import get_cnn_embeddings

# fix random seed for reproducibility
np.random.seed(21)


class CNN_model:

    def __init__(self,
                 input_size=50,
                 output_size=3,
                 loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'],
                 random_seed=87):
        self.input_size = input_size     # tokens
        self.output_size = output_size

        np.random.seed(random_seed)

        self.model = Sequential()
        self.model.add(Conv1D(filters=150, kernel_size=3,
                              padding='same', activation='relu',
                              input_shape=(input_size, 400)))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(250, activation='relu'))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X_train, y_train,
            batch_size=32,
            epochs=3,
            validation_split=0.2,
            **kwargs):
        self.model.fit(X_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split,
                       **kwargs)

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose)

    def evaluate_model(self, y_test, y_pred):
        # output evaluation data
        print('--------- INDIVIDUAL CNN Model results ---------')
        print('--------- F-1/precision/recall report  ---------')
        print('---------         MACRO F1             ---------')
        print(f1_score(y_test, y_pred, average='macro'))
        print('---------         F1 Matrix            ---------')
        print(evaluation.evaluate_results(y_test, y_pred))

    def get_model_summary(self):
        return self.model.summary()

    def save_model(self, save_model_path):
        self.model.save(save_model_path)  # creates a HDF5 file 'my_model.h5'

    def load_model(self, load_model_path):
        self.model = load_model(load_model_path)


class Ensemble_CNN_model:

    def __init__(self,
                 load_weight_path=None,
                 num_ensemble_models=10,
                 input_size=50,
                 output_size=3,
                 random_seed=87):

        self.ensemble_cnns = [None] * num_ensemble_models

        for i in range(num_ensemble_models):
            np.random.seed(int(random_seed * (i + 1)))
            self.ensemble_cnns[i] = CNN_model(input_size=input_size,
                                              output_size=output_size)
            if load_weight_path:
                try:
                    self.ensemble_cnns[i].load_model(
                        load_weight_path + f'_{i}')
                except:
                    print(
                        f'Model weight not available: {load_weight_path}_{i}')

    def predict(self, X):
        aggregate_y_pred = []
        encoder = None

        RANGE = len(self.ensemble_cnns)
        for cnn_model in self.ensemble_cnns:
            y_soft_max = cnn_model.predict(X)
            y_pred = y_soft_max.argmax(axis=1)
            aggregate_y_pred.append(y_soft_max)

        # ADD ALL RESULTS
        sum_of_ys = aggregate_y_pred[0]
        for i in [x + 1 for x in range(RANGE - 1)]:
            sum_of_ys += aggregate_y_pred[i]

        # DIVIDE BY RANGE FOR MEAN
        sum_of_ys /= RANGE

        # ENCODE PREDS
        encoded_preds = sum_of_ys.argmax(axis=1)
        print(len(sum_of_ys), len(encoded_preds))
        return encoded_preds

    def fit_and_eval(self, X_train, y_train, X_test, y_test,
                     save_weight_path=None,
                     validation_split=0.2,
                     batch_size=32,
                     epochs=3,
                     **kwargs):
        aggregate_y_pred = []
        encoder = None

        RANGE = len(self.ensemble_cnns)
        for cnn_model in self.ensemble_cnns:
            y_encoder, y_one_hot = helper.one_hot_encode_y(y_train)
            cnn_model.fit(X_train, y_one_hot, epochs=epochs,
                          batch_size=batch_size)

            if save_weight_path:
                cnn_model.save_model(save_weight_path)

            y_soft_max = cnn_model.predict(X_test)
            encoded_preds = y_soft_max.argmax(axis=1)
            decoded_preds = y_encoder.inverse_transform(encoded_preds)

            cnn_model.evaluate_model(
                y_test, decoded_preds)  # print indv results
            aggregate_y_pred.append(y_soft_max)

        # ADD ALL RESULTS
        sum_of_ys = aggregate_y_pred[0]
        for i in [x + 1 for x in range(RANGE - 1)]:
            sum_of_ys += aggregate_y_pred[i]

        # DIVIDE BY RANGE FOR MEAN
        sum_of_ys /= RANGE

        # ENCODE PREDS
        encoded_preds = sum_of_ys.argmax(axis=1)
        decoded_preds = y_encoder.inverse_transform(encoded_preds)
        print(len(sum_of_ys), len(encoded_preds), len(decoded_preds))
        self.evaluate_model(y_test, decoded_preds)

    def evaluate_model(self, y_test, y_pred):
        print('--------- FINAL ENSEMBLE Model results ---------')
        print('---------  F-1/precision/recall report ---------')
        print('---------            MACRO F1:         ---------')
        print(f1_score(y_test, y_pred, average='macro'))
        print('---------            F1 Matrix         ---------')
        print(evaluation.evaluate_results(y_test, y_pred))
