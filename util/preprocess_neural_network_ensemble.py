import numpy as np


def get_cnn_embeddings(word_embeddings, tweets_tokenized, max_tokens=50):
    '''twitter embedding model only'''
    corpus_vecs = []
    for tweet in tweets_tokenized:
        tweet_vecs = [[0.0 for x in range(400)] for x in range(max_tokens)]
        for cnt, token in enumerate(tweet):
            try:
                tweet_vecs[cnt] = (word_embeddings[token].tolist())
            except:
                continue
        # tweet_vecs.append(embedding_sum/tweet_length)  # NOTE: L1 and High C 10+  better for this scenario
        # NOTE: L2 and Low C .01- better for this scenario
        corpus_vecs.append(tweet_vecs)
    return np.array(corpus_vecs)


def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "Hate speech"
    elif class_label == 1:
        return "Offensive language"
    elif class_label == 2:
        return "Neither"
    else:
        return "No label"
