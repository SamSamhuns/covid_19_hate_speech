import matplotlib.pyplot as plt
import re
import sys
import nltk
import string
import seaborn
import numpy as np
import pandas as pd
from tqdm import tqdm


import textstat
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]  # add retweeted status to exclusions
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


def tokenize(tweet, tokenizer_wlen_thres=300):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()
              if len(t) < tokenizer_wlen_thres]
    return tokens


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet, sentiment_analyzer):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    # Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) +
                 float(11.8 * avg_syl) - 15.59, 1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) -
                (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    return features


def get_feature_array(tweets):
    feats = []
    sentiment_analyzer = VS()

    for t in tweets:
        feats.append(other_features(t, sentiment_analyzer))
    return np.array(feats)


def generate_features_train_data(tweets):
    """
    tweet_list must be a series or a list
    Returns preprocessed_tweets, vectorizer and pos_vectorizer
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,
        use_idf=True,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75
    )

    # We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.75,
    )

    # Construct tfidf matrix and get relevant scores
    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    # keys are indices; values are IDF scores
    idf_dict = {i: idf_vals[i] for i in vocab.values()}

    # Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tqdm(tweets):
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    # Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v: i for i, v in enumerate(
        pos_vectorizer.get_feature_names())}

    # Now get other features
    feats = get_feature_array(tweets)

    # Now join them all up
    concatenated_feats = np.concatenate([tfidf, pos, feats], axis=1)

    return concatenated_feats, vectorizer, pos_vectorizer


def generate_features_test_data(tweet_list, vectorizer, pos_vectorizer):
    """
    tweet_list must be a series or a list
    """

    # Construct tfidf matrix and get relevant scores
    tfidf = vectorizer.transform(tweet_list.values.astype('U')).toarray()
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    # keys are indices; values are IDF scores
    idf_dict = {i: idf_vals[i] for i in vocab.values()}

    # Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tqdm(tweet_list):
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    # Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v: i for i, v in enumerate(
        pos_vectorizer.get_feature_names())}

    # Now get other features
    feats = get_feature_array(tweet_list)

    # Concatenate feats and return
    return pd.DataFrame(np.concatenate([tfidf, pos, feats], axis=1))
