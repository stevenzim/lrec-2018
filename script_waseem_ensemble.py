from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.model_selection import train_test_split


from src import nlp, helper, evaluation

from src import evaluation



REPORT_FILE_NAME = 'resources/results/waseem_evaluation_ensemble.results'

# DETERMINES IF EMBEDDING VECS ARE AVERAGED OR SUMMED FOR DOCUMENT
SUM_OR_AVG = 'sum'



#
# -------EMBEDDING MODELS -------------

# Load tweet embeddings lookup
# ----- GODIN ------------
LOWER_CASE_TOKENS = False
word_vectors = KeyedVectors.load_word2vec_format('resources/pre_trained_models/word2vec_twitter_model.bin', binary=True,
                                                 unicode_errors='ignore')

# ----- WASEEM HOVY DATA (16208 TOTAL) -------


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



hate_corpus = helper.load_json_from_file('resources/hate_speech_corps/NAACL_SRW_2016_DOWNLOADED.json')

print ('Extracting features')
X_train = get_cnn_embeddings(word_vectors,
                             map(lambda y: nlp.replace_tokens(y),
                                 nlp.tokenize_tweets(map(lambda x: x['text'], hate_corpus),
                                                     lower_case=LOWER_CASE_TOKENS)), max_tokens=50)
Y_train = map(lambda x: x['class'], hate_corpus)


print '-----CNN    -----------'
from sklearn.model_selection import train_test_split

X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(
    X_train, Y_train, test_size=0.15, random_state=7)



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
    from keras import initializers
    seed = random_seed
    np.random.seed(seed)
    model = Sequential()
    # create model
    model.add(Conv1D(filters=150, kernel_size=3, padding='same', activation='relu', input_shape=(tokens, 400)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def waseem_hovy_cnn(epochs=5, batch_size=50, tokens=50, random_seed= 21):


    Y_hate_encoder, one_hot_y_hate, encoded_Y_hate = helper.encode_ys_categorical(Y_train_split)

    # BUILD MODEL
    # np.random.seed(random_seed)

    m = cnn_model(tokens=tokens, random_seed=random_seed)
    m.fit(X_train_split, one_hot_y_hate, epochs=epochs, batch_size=batch_size)

    # Get Predictions
    c = m.predict(np.array(X_test_split))
    encoded_preds = c.argmax(axis=1)
    decoded_preds = Y_hate_encoder.inverse_transform(encoded_preds)

    # output evaluation data
    print '----------F-1/precision/recall report-----------'
    print 'MACRO F1:'
    print f1_score(Y_test_split, decoded_preds, average='macro')
    print 'F1 Matrix'
    print evaluation.evaluate_results(Y_test_split, decoded_preds)
    o_file = open(REPORT_FILE_NAME, 'a')
    o_file.write(str(f1_score(Y_test_split, decoded_preds, average='macro')) + '\n')
    o_file.close()

    # first generate with specified labels
    labels = ['none', 'racism', 'sexism']
    cm = confusion_matrix(Y_test_split, decoded_preds, labels)

    # then print it in a pretty way
    print '-------confusion matrix---------'
    print evaluation.print_cm(cm, labels)

    Y_soft_max = c
    return Y_soft_max , Y_hate_encoder, Y_test_split


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
        Y_soft_max, Y_hate_encoder, Y_test_split = waseem_hovy_cnn(epochs=epochs, batch_size=batch_size, random_seed=int(random_int*(i+1)))
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

    # o_file.write('ensemble,')
    # o_file.write(str(f1_score(Y_test_2013, decoded_preds, average='macro')) + ',')
    o_file = open(REPORT_FILE_NAME, 'a')
    o_file.write('waseem----ensemble--->\n')
    o_file.write(str(f1_score(test_split, decoded_preds, average='macro')) + '\n')
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










