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


class CombinedMiniEnsemble:

    def __init__(self, logreg_model,
                 svc_model):
        self.logreg_model = logreg_model
        self.svc_model = svc_model

    def predict(self, X, vectorizer, pos_vectorizer):
        X_m_1_2 = generate_features_test_data(X, vectorizer, pos_vectorizer)

        y_m1 = self.logreg_model.predict(X_m_1_2)
        y_m2 = self.svc_model.predict(X_m_1_2)

        return np.array([np.bincount(arr).argmax() for arr in np.vstack((y_m1, y_m2)).T])


class CombinedEnsemble:

    def __init__(self, nn_ensemble_model,
                 logreg_model,
                 svc_model):
        self.nn_ensemble_model = nn_ensemble_model
        self.logreg_model = logreg_model
        self.svc_model = svc_model

    def predict(self, X, word_vectors, vectorizer, pos_vectorizer):
        X_m1 = get_cnn_embeddings(word_vectors,
                                  map(lambda y: nlp.replace_tokens(y),
                                      nlp.tokenize_tweets(X,
                                                          lower_case=LOWER_CASE_TOKENS)), max_tokens=50)
        X_m_2_3 = generate_features_test_data(X, vectorizer, pos_vectorizer)

        y_m1 = self.nn_ensemble_model.predict(X_m1)
        y_m2 = self.logreg_model.predict(X_m_2_3)
        y_m3 = self.svc_model.predict(X_m_2_3)

        return np.array([np.bincount(arr).argmax() for arr in np.vstack((y_m1, y_m2, y_m3)).T])
