# coding=utf-8
"""Basic NLP Stuff to prep tweets and other text for embedding models"""

import re

from nltk.tokenize import TweetTokenizer
import numpy as np


def tokenize_tweets(list_of_tweets, lower_case=False):
    tknzr = TweetTokenizer()

    tweets_tokenized = []
    for tweet in list_of_tweets:
        if lower_case:
            tweet = tweet.lower()
        tweets_tokenized.append(tknzr.tokenize(tweet))
    return tweets_tokenized


def replace_tokens(list_of_tokens):
    """
    Per instructions from Fréderic Godin at Ghent University
    links with _URL_, anything with digits to _NUMBER_ and usernames to _MENTION_ 
    """
    for i, token in enumerate(list_of_tokens):
        if ('http:' in token) or ('https:' in token):
            list_of_tokens[i] = '_URL_'
            continue
        if '@' == token[0]:
            list_of_tokens[i] = '_MENTION_'
            continue
        if bool(re.search(r'\d', token)):
            list_of_tokens[i] = '_NUMBER_'
            continue
    return list_of_tokens


# Test case
tokes = ['@steve', 'http://steve.com', 'https://steve.com', '100%', '521', 'test']
replace_tokens(tokes)


def load_glove_model(glove_file):
    print "Loading Glove Model"
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

def get_embeddings(word_embedding_lookup, tokenized_docs, sum_or_avg ='sum'):
    """
    :param word_embedding_lookup: Gensim type Keyed Vector embedding lookup model
    :param tokenized_docs: set of docs with text tokenized in list form e.g. [['i', 'love', 'you'],['i', 'hate', 'you']
    :param sum_or_avg: 'sum' = return a sum 'avg' = return avg of all embeddings for document
    :return: return embedding scores for document
    """
    '''twitter embedding model only'''
    try:
        # type is word_2_vec from gensim
        embedding_dim = word_embedding_lookup.vector_size
    except:
        # type is GloVe embedding
        embedding_dim = len(word_embedding_lookup[word_embedding_lookup.keys()[0]])
    embedding_vecs = []
    for tweet in tokenized_docs:
        embedding_sum = np.zeros(embedding_dim)
        for token in tweet:
            try:
                embedding_sum += word_embedding_lookup[token]
            except:
                continue
        if sum_or_avg == 'avg':
            embedding_vecs.append(embedding_sum / len(tokenized_docs))  # NOTE: L1 and High C 10+  better for this scenario
        else:
            embedding_vecs.append(embedding_sum)  # NOTE: L2 and Low C .01- better for this scenario
    return embedding_vecs
