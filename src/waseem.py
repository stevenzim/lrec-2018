'''Function to load Waseem/Hovy tweets set 1 and classes from csv and return lists of data'''
# Load datasets
import csv

# shuffle data (the tsv files are grouped by racism, sexism and none)
import random
from random import shuffle

random.seed(21)

def get_hate_data():
    X_hate_raw = []
    Y_hate_raw = []
    with open('resources/hate_speech_corps/NAACL_SRW_2016_DOWNLOADED.csv', 'rb') as tweet_tsv:
        tweet_tsv = csv.reader(tweet_tsv, delimiter='\t')

        for row in tweet_tsv:
            X_hate_raw.append(row[2])
            Y_hate_raw.append(row[1])


    randomised_hate_idxs = range(len(X_hate_raw))
    shuffle(randomised_hate_idxs)

    X_hate = []
    Y_hate = []
    Y_hate_binary = []

    for idx in randomised_hate_idxs:
        # we don't want tweets that were not available (~700 tweets of 16,883 set were not available for download)
        if X_hate_raw[idx] == 'Not Available':
            continue
        else:
            # create a multiclass set and binary set
            X_hate.append(X_hate_raw[idx])
            Y_hate.append(Y_hate_raw[idx])
            if Y_hate_raw[idx] == 'none':
                Y_hate_binary.append(False)
            else:
                Y_hate_binary.append(True)

    return X_hate, Y_hate
