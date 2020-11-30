import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

# based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

numpy.random.seed(7)

top_words = 5000
max_review_length = 500


def load_data():
    # load the dataset but only keep the top n words, zero the rest
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    return (X_train, y_train), (X_test, y_test)


def preproces_data(X_train, X_test):
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    return X_train, X_test


def build_classifier(top_words, X_train, y_train, X_test, y_test, max_review_length):
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
    return model, history


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


def main():
    (X_train, y_train), (X_test, y_test) = load_data()
    (X_train, X_test) = preproces_data(X_train, X_test)
    model, history = build_classifier(top_words, X_train, y_train, X_test, y_test, max_review_length)
    evaluate(model, X_test, y_test)
    render_plot(history)


if __name__ == '__main__':
    main()
