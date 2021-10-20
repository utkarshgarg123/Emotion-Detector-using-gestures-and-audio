"""
This example demonstrates how to use `LSTM` model from
`speechemotionrecognition` package
"""

from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


def lstm_example(speech_data):
    to_flatten = False
    x_train, y_train, x_test, y_test, num_labels = extract_data(speech_data)
    y_train = np_utils.to_categorical(y_train)
    model = LSTM(input=x_train[0].shape,
                 num_classes=num_labels)
    model.evaluate(x_test, y_test)
    return('prediction', model.predict_one(
        get_feature_vector_from_mfcc(x_train, flatten=to_flatten)),
          'Actual 3')


if __name__ == '__main__':
    lstm_example()
