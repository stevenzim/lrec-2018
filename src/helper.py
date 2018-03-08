import json

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



def load_json_from_file(file_name):
    '''Loads JSON data into list.  Returns a list of dictionaries in [{},{}...{}] format
    fileName is relative to directory where Python is running.
    usage: myListOfDicts = load_json_from_file("FileName") '''
    #return json.loads(unicode(open(fileName).read(), "ISO-8859-1"))
    return json.loads(open(file_name).read())

def dump_json_to_file(file_name, data_to_dump):
    """Dumps JSON data from dictionaries in list into output file. This data is pretty printed
    for readability. fileName is relative to directory where Python is running.
    dataToDump should be in [{},{}...{}] format
    usage: dumpJSONtoFile("FileName", myListOfDicts)"""
    with open(file_name, 'w') as outfile:
        json.dump(data_to_dump, outfile, sort_keys=True, indent=4, separators=(',', ': '))



def encode_ys_categorical(label_array):
    """
    :param label_array:  1D array of binary or multiclass labels (type int or string)
    :return: numpy encoded version of array && one hot matrix necessary for keras inputs
    """
    encoder = LabelEncoder()
    encoder.fit(label_array)
    encoded_Y = encoder.transform(label_array)

    # convert integers to dummy variables (i.e. one hot encoded)
    one_hot_y = np_utils.to_categorical(encoded_Y)
    return encoder, one_hot_y, encoded_Y