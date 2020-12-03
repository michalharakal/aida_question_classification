import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.backend import clear_session
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model
import datetime
import pickle
import glob
import os

# Model parameters
LSTM_SIZE = 100
DENSE_SIZE = 100
batch_size = 128
epochs = 20

numpy.random.seed(7)


def model_name_2_file_name(model_name, basedir="./"):
    """ Return model file name from model name, contain relative path """
    return os.path.join(basedir, "models", model_name + ".h5")


def model_name_2_plot_name(model_name, basedir="./"):
    """ Return image file name from model name, contain relative path """
    return os.path.join(basedir, "models", model_name + ".png")


def new_model_needed(model_name):
    """
    Check if trained model already exists.
    :return: true if a new model has to be created
    """
    return os.path.isfile(model_name_2_file_name(model_name)) is False


def load_local_model(model_name: str, basedir="./"):
    """
    Load model from the file.
    :return: Model. Throws error if file not found
    """
    file_name = model_name_2_file_name(model_name, basedir)
    return load_model(file_name)


def save_model_locally(model, basedir="./"):
    """ Save trained model locally in Keras format"""
    file_name = model_name_2_file_name(model.name, basedir)
    save_model(model, file_name)


def save_history_locally(model, history, basedir="./"):
    """ Save history. Add timestamp to file name"""
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    filename = f"{model.name}-{timestamp}.pkl"
    with open(os.path.join(basedir, "report", filename), 'wb') as pickle_out:
        pickle.dump(history.history, pickle_out)


def load_latest_local_history(model, basedir="./"):
    """ iterate over local folder and return the latest history file for given model"""
    list_of_files = glob.glob(f"{basedir}/report/{model.name}*.pkl")  # * means all if need specific format then *.csv
    if list_of_files is None or len(list_of_files) == 0:
        return None
    latest_file_name = max(list_of_files, key=os.path.getctime)
    with open(latest_file_name, 'rb') as fid:
        return pickle.load(fid)


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
    latest_local_history = load_latest_local_history(model)
    if new_model_needed(model.name) or latest_local_history is None:
        history = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.1)
        save_model_locally(model)
        save_history_locally(model, history)
        return history.history
    else:
        return latest_local_history
