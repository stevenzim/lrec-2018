from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.model_selection import train_test_split


from src import nlp, helper, evaluation


from src import evaluation


REPORT_FILE_NAME = 'resources/results/semeval_evaluation_online_train.results'

# DETERMINES IF EMBEDDING VECS ARE AVERAGED OR SUMMED FOR DOCUMENT
SUM_OR_AVG = 'sum'



#
# -------EMBEDDING MODELS -------------

# Load tweet embeddings lookup
# ----- GODIN ------------
LOWER_CASE_TOKENS = False
word_vectors = KeyedVectors.load_word2vec_format('resources/pre_trained_models/word2vec_twitter_model.bin', binary=True,
                                                 unicode_errors='ignore')
'''DATA'''

def get_cnn_embeddings(word_embeddings, tweets_tokenized, max_tokens = 50):
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
        corpus_vecs.append(tweet_vecs)  # NOTE: L2 and Low C .01- better for this scenario
    return corpus_vecs

# ----- TRAIN + DEV DATA SETS (SIZE 9649 + 1639)-----
train_data = helper.load_json_from_file('resources/sem_eval_tweets/SemEval/SemTrain.json')
print len(train_data)
train_data.extend(helper.load_json_from_file('resources/sem_eval_tweets/SemEval/SemDev.json'))
print len(train_data)

X_train = get_cnn_embeddings(word_vectors,
                             map(lambda y: nlp.replace_tokens(y), nlp.tokenize_tweets(map(lambda x: x['text'], train_data), lower_case=LOWER_CASE_TOKENS)))
Y_train = map(lambda x: x['sentiment_num'], train_data)


# ----- 2013 DATA SETS (SIZE 3797 ) -----
test_data_2013 = helper.load_json_from_file('resources/sem_eval_tweets/SemEval/SemTest2013.json')
X_test_2013 = get_cnn_embeddings(word_vectors,
                                 map(lambda y: nlp.replace_tokens(y), nlp.tokenize_tweets(map(lambda x: x['text'], test_data_2013), lower_case=LOWER_CASE_TOKENS)))
Y_test_2013 = map(lambda x: x['sentiment_num'], test_data_2013)





'''
Hate speech with CNN's 
'''
# CNN for the IMDB problem
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D

import numpy as np

# fix random seed for reproducibility
seed = 21
np.random.seed(seed)




# create the model
def cnn_model(tokens = 50, random_seed = 21):
    seed = random_seed
    np.random.seed(seed)
    model = Sequential()
    # create model
    model.add(Conv1D(filters=150, kernel_size=3, activation="relu", padding='same',input_shape=(tokens, 400)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def semeval_cnn(epochs=5, batch_size=50, tokens=50, random_seed=21):


    # SEMEVAL GRID / PARAMETER SEARCH
    from sklearn.model_selection import GridSearchCV

    print '-----CNN    -----------'

    Y_hate_encoder, one_hot_y_hate, encoded_Y_hate = helper.encode_ys_categorical(Y_train)

    # BUILD MODEL
    # np.random.seed(random_seed)

    m = cnn_model(tokens=tokens, random_seed=random_seed)
    # Y_encoder, one_hot_y = encode_ys_categorical(Y_train)
    m.fit(X_train, one_hot_y_hate, epochs=epochs, batch_size=batch_size)

    def prediction(model, X_set = X_test_2013, Y_set = Y_test_2013, test_set='--------2013 set-------'):
        print (test_set)
        # Get Predictions
        c = model.predict(np.array(X_set))
        encoded_preds = c.argmax(axis=1)
        decoded_preds = Y_hate_encoder.inverse_transform(encoded_preds)

        print '----------F-1/precision/recall report-----------'
        print 'MACRO F1:'
        print f1_score(Y_test_2013, decoded_preds, average='macro')
        print 'F1 Matrix'
        print evaluation.evaluate_results(Y_set, decoded_preds)
        # o_file.write(str(random_seed) + ',')
        # o_file.write(str(f1_score(Y_test_2013, decoded_preds, average='macro')) + ',')
        o_file = open(REPORT_FILE_NAME, 'a')
        o_file.write(str(evaluation.sem_eval_f1_avg(Y_set, decoded_preds)) + '\n')
        o_file.close()


        # first generate with specified labels
        labels = ['negative', 'neutral', 'positive']
        cm = confusion_matrix(Y_set, decoded_preds)

        # then print it in a pretty way
        print '-------confusion matrix---------'
        print evaluation.print_cm(cm, labels)
        return c

    Y_soft_max_2013 = prediction(m, X_set = X_test_2013, Y_set = Y_test_2013, test_set='--------2013 set-------')
    # Y_soft_max_2014 = prediction(m, X_set = X_test_2014, Y_set = Y_test_2014, test_set='--------2014 set-------')
    # Y_soft_max_2015 = prediction(m, X_set = X_test_2015, Y_set = Y_test_2015, test_set='--------2015 set-------')
    return Y_soft_max_2013 , Y_hate_encoder, Y_test_2013


def run_ensemble(epochs = 5, batch_size=50, random_int=100):
    o_file = open(REPORT_FILE_NAME, 'a')
    o_file.write(str(random_int)  + '\n')
    o_file.write(str(epochs) + '\n')
    o_file.write(str(batch_size) + '\n')
    o_file.close()
    y_s_list = []
    encoder = None
    test_split = None
    RANGE = 10
    for i in list(range(RANGE)):
        Y_soft_max, Y_hate_encoder, Y_test_split = semeval_cnn(epochs=epochs, batch_size=batch_size, random_seed=int(random_int*(i+1)))
        y_s_list.append(Y_soft_max)
        encoder = Y_hate_encoder
        test_split = Y_test_split



    # ADDD ALL RESULTS
    sum_of_ys = y_s_list[0]
    for i in [x + 1 for x in range(RANGE - 1)]:
        sum_of_ys += y_s_list[i]

    # DIVIDE BY RANGE FOR MEAN
    sum_of_ys /= RANGE

    # ENCODE PREDS
    encoded_preds = sum_of_ys.argmax(axis=1)


    decoded_preds = encoder.inverse_transform(encoded_preds)

    print '----------F-1/precision/recall report-----------'
    print 'MACRO F1:'
    print f1_score(test_split, decoded_preds, average='macro')
    print 'F1 Matrix'
    print evaluation.evaluate_results(test_split, decoded_preds)
    print evaluation.sem_eval_f1_avg(test_split, decoded_preds)

    # o_file.write('ensemble,')
    # o_file.write(str(f1_score(Y_test_2013, decoded_preds, average='macro')) + ',')
    o_file = open(REPORT_FILE_NAME, 'a')
    o_file.write('2013----ensemble--->\n')
    o_file.write(str(evaluation.sem_eval_f1_avg(test_split, decoded_preds)) + '\n')
    o_file.write('$$$$$$$$$$$$$$$$  END $$$$$$$$$$$$$$$$$$$$$$$$$\n')
    o_file.close()


batch_size = [10, 25, 50, 100]
epochs = [3, 5, 10]
random_int = [87, 100, 101]


for bs in batch_size:
    for epoch in epochs:
        for i in random_int:
            o_file = open(REPORT_FILE_NAME, 'a')
            o_file.write('$$$$$$$$$$$$$$$$  START $$$$$$$$$$$$$$$$$$$$$$$$$\n')
            run_ensemble(epochs = epoch, batch_size=bs, random_int=i)
            o_file.close()













