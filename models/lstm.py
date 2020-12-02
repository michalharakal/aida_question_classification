import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

# based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

numpy.random.seed(7)


def build_classifier(vocab_size, seq_length, LSTM_SIZE, DENSE_SIZE):
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, seq_length, input_length=seq_length, name="Embedding"))
    model.add(LSTM(LSTM_SIZE, return_sequences=True, name="LSTM1"))
    model.add(LSTM(LSTM_SIZE, name="LSTM2"))
    model.add(Dense(DENSE_SIZE,  activation='relu', name="DenseRelu"))
    model.add(Dense(1, activation='softmax', name="DenseSoftmax"))
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


def render_plot(history):
    plt.figure(figsize=(15, 5))
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['val_accuracy'], label='Test accuracy')

    plt.xticks(range(3))
    plt.title("Model", fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig('lstm_history.png')


def archive_results(score):
    pass
