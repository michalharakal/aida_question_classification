import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Model parameters
LSTM_SIZE = 100
DENSE_SIZE = 100
batch_size = 128
epochs = 20

numpy.random.seed(7)


def build_classifier_simple(vocab_size, seq_length, categories_count):
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, seq_length, input_length=seq_length, name="Embedding"))
    model.add(LSTM(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(categories_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def build_classifier(vocab_size, seq_length, categories_count):
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, seq_length, input_length=seq_length, name="Embedding"))
    model.add(LSTM(LSTM_SIZE, return_sequences=True, name="LSTM1"))
    model.add(LSTM(LSTM_SIZE, name="LSTM2"))
    model.add(Dense(DENSE_SIZE, activation='relu', name="Denserelu"))
    model.add(Dense(categories_count, activation='softmax', name="DenseSoftmax"))
    model.compile(loss='categorical_crossentropy', \
                  optimizer='adam', \
                  metrics=['accuracy'])
    print(model.summary())
    return model


def evaluate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return scores


def train(model, X_train, Y_train):
    # Ativando o Tensorboard para poder monitorar com o comando abaixo em uym outro terminal:
    # > tensorboard --logdir=logskeras/
    tensorboard = TensorBoard(log_dir='logskeras/{}'.format(datetime.datetime.today().strftime('%Y%m%d_%H%M')))
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard])
    return history
