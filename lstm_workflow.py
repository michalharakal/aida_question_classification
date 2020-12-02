import data.get_data as data
import data.preprocess as dp
import models.lstm as lstm
import matplotlib.pyplot as plt


# %%

def evaluate(model, X_test, y_test):
    pass


def render_plot(history):
    plt.figure(figsize=(4, 5))
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.plot(history.history['loss'], label='Test accuracy')

    plt.title("Title", fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig("plots/lstm1.png")
    pass


# Model parameters
LSTM_SIZE = 100
DENSE_SIZE = 100
batch_size = 128
epochs = 20


def main():
    test_df = data.get_test_data()
    train_df = data.get_train_data()

    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = dp.preprocess_data(train_df, test_df)

    model = lstm.build_classifier(vocab_size, LSTM_SIZE, DENSE_SIZE, sequence_length)
    # compile model

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    evaluate(model, X_test, y_test)
    render_plot(history)


if __name__ == '__main__':
    main()
