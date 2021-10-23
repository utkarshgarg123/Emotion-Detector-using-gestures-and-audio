"""
This example demonstrates how to use `LSTM` model from
`speechemotionrecognition` package
"""

from keras.utils import np_utils
import os

from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
from tensorflow.keras.models import load_model
MODEL_PATH = os.getcwd()+"..\\models\\best_model_LSTM.h5"
def lstm_example(speech_data):
    model = load_model(MODEL_PATH)
    return( model.predict_one(
        get_feature_vector_from_mfcc(speech_data, flatten=False)))


if __name__ == '__main__':
    lstm_example()
