import os

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import TensorBoard
from keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import datetime
import pydot

# Model parameters
LSTM_SIZE = 100
DENSE_SIZE = 100
batch_size = 128
epochs = 20

numpy.random.seed(7)


def model_name_2_file_name(model_name):
    return os.path.join("./models", model_name + ".h5")


def model_name_2_plot_name(model_name):
    return os.path.join("./models", model_name + ".png")


def new_model_needed(model_name):
    return os.path.isfile(model_name_2_file_name(model_name)) is False


def load_local_model(model_name):
    file_name = model_name_2_file_name(model_name)
    return load_model(file_name)


def plot_model_summary(model):
    plot_model(model, to_file=model_name_2_plot_name(model.name), show_shapes=True, show_layer_names=True)


def build_classifier_lstm_dropout(vocab_size, seq_length, categories_count, model_name="LSTM_DROPOUT") -> object:
    if new_model_needed(model_name):
        clear_session()

        model = Sequential(name=model_name)
        model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=seq_length))
        model.add(LSTM(units=128))
        model.add(Dropout(0.5))
        model.add(Dense(categories_count, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        return model
    else:
        return load_local_model(model_name)


def build_classifier_lstm3(vocab_size, seq_length, categories_count, model_name="LSTM3"):
    if new_model_needed(model_name):
        clear_session()

        model = Sequential(name=model_name)
        model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=seq_length))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(categories_count, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        return model
    else:
        return load_local_model(model_name)


def build_classifier_lstm2_dense2(vocab_size, seq_length, categories_count, model_name="LSTM2_DENSE_2"):
    if new_model_needed(model_name):
        clear_session()
        # create the model
        model = Sequential(name=model_name)
        model.add(Embedding(vocab_size, seq_length, input_length=seq_length, name="Embedding"))
        model.add(LSTM(LSTM_SIZE, return_sequences=True, name="LSTM1"))
        model.add(LSTM(LSTM_SIZE, name="LSTM2"))
        model.add(Dense(DENSE_SIZE, activation='relu', name="DenseRelu"))
        model.add(Dense(categories_count, activation='softmax', name="DenseSoftmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model
    else:
        return load_local_model(model_name)


def evaluate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return scores


def train(model, X_train, Y_train):
    tensorboard = TensorBoard(log_dir='logskeras/{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M')))

    history = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.1,
                        callbacks=[tensorboard])
    return history
