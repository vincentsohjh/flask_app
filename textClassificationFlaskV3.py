
import sys
sys.path = [
    r'C:\Users\HuaSheng\Anaconda3\envs\tfkrjn\Lib\site-packages'] + [''] + sys.path

import pickle
import keras

import numpy as np
from flask import Flask, abort, jsonify, request
from keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import tensorflow as tf
import re

from keras.models import load_model
from keras.layers import *


# skip this when in ipynb
app = Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    # all kinds of error checking should go here
    print(keras.__version__)
    clear_session()
    with tf.device('/cpu:0'):
        financeMdl = load_model('./textClassWithActionsWithClosest.h5')
    financeMdl.summary()
    data = request.get_json(force=True)
    textIn = [data['dataIn']]
    print(textIn)

    # keep also %$ but removed comma
    textIn = re.sub(r"[^A-Za-z0-9()!?\'\`%$]", " ", textIn[0])
    textIn = re.sub(r"\'s", " \'s", textIn)
    textIn = re.sub(r"\'ve", " \'ve", textIn)
    textIn = re.sub(r"n\'t", " n\'t", textIn)
    textIn = re.sub(r"\'re", " \'re", textIn)
    textIn = re.sub(r"\'d", " \'d", textIn)
    textIn = re.sub(r"\'ll", " \'ll", textIn)
    textIn = re.sub(r"!", " ! ", textIn)
    textIn = re.sub(r"\(", " ( ", textIn)
    textIn = re.sub(r"\)", " ) ", textIn)
    textIn = re.sub(r"\?", " ? ", textIn)
    textIn = re.sub(r"\$", " $ ", textIn)  # yes, isolate $
    textIn = re.sub(r"\%", " % ", textIn)  # yes, isolate %
    textIn = re.sub(r"\s{2,}", " ", textIn)
    # fixing XXX and xxx like as word
    textIn = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", textIn)
    # removing non ascii
    textIn = re.sub(r'[^\x00-\x7F]+', "", textIn)
    textIn = textIn.strip().lower()

    MAX_SEQUENCE_LENGTH = 80
    word_data = []
    word_data.append(textIn)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    sequencestr1 = tokenizer.texts_to_sequences(word_data)

    datatr1 = pad_sequences(sequencestr1, maxlen=MAX_SEQUENCE_LENGTH)

    predict_request = datatr1
    #[data['datatr1']]
    predict_request = np.array(predict_request).reshape(1, -1)
    print(predict_request.shape)
    y_hat = financeMdl.predict(predict_request)
    inv_map, inv_map1, inv_map2, inv_map3 = pickle.load(open('objs.pkl', 'rb'))
    refVect, refContent = pickle.load(open('closestSel.pkl', 'rb'))
    diffVect = np.sum(np.abs(refVect - y_hat[4]), axis=1)
    retStr = []
    for i in range(5):
        retStr += 'case ID ' + \
            str(np.argsort(diffVect)[i]) + ' --> ' + \
            refContent[np.argsort(diffVect)[i]]
    retStr = ''.join(retStr)
    # return our prediction
    print(y_hat)
    textUser = 'Input text classified as ' + inv_map[np.argmax(y_hat[0])] + \
        '. It is ' + inv_map2[np.argmax(y_hat[2])] + '. Recommend to send to '\
        + inv_map1[np.argmax(y_hat[1])] + ' department. Proposed Response: ' + \
        inv_map3[np.argmax(y_hat[3])]
    output = {'y_hat': inv_map[np.argmax(y_hat[0])], 'y_hat1': inv_map1[np.argmax(y_hat[1])],
              'y_hat2': inv_map2[np.argmax(y_hat[2])], 'y_hat3': inv_map3[np.argmax(y_hat[3])], 'y_hat4': textUser,
              'y_hat5': retStr}
    # output = {'y_hat': y_hat[0].tolist(),'y_hat1': y_hat[1].tolist(),
    #          'y_hat2': y_hat[2].tolist(),'y_hat3': y_hat[3].tolist()}
    print(output)
    return jsonify(results=output)


@app.route('/process', methods=['POST'])
def clean_str(string):
    """
    Tokenization/string cleaning (partially modified)
    """
    string = re.sub(r"[^A-Za-z0-9()!?\'\`%$]", " ",
                    string)  # keep also %$ but removed comma
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\$", " $ ", string)  # yes, isolate $
    string = re.sub(r"\%", " % ", string)  # yes, isolate %
    string = re.sub(r"\s{2,}", " ", string)

    # fixing XXX and xxx like as word
    string = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", string)
    # removing non ascii
    string = re.sub(r'[^\x00-\x7F]+', "", string)

    return string.strip().lower()


if __name__ == '__main__':
    print('staret')
    app.run(port=9000, debug=True)


# In[ ]:
