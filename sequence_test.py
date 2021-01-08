import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def CNN_imdb_Test():
    data=keras.datasets.imdb
    (x_train,y_train),(x_test,y_test)=data.load_data()