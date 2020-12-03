import data.get_data as data
import data.preprocess as dp
from models.lstm import lstm_model as lstm_model
from models.lstm import evaluate as lstm_evaluate
import utils.tf_utils  as tf_utils


def main():
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = dp.preprocess_data(train_df, test_df)
    # build and train model
    model = lstm_model.build_classifier_lstm(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    lstm_evaluate.evaluate(model, X_test, y_test)
    lstm_evaluate.render_plot(history)


if __name__ == '__main__':
    tf_utils.activate_compatibility_allow_growth()
    main()
