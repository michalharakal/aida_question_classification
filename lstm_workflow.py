import data.get_data as data
import data.preprocess as dp
from models.lstm import lstm_model as lstm_model
from models.lstm import evaluate as lstm_evaluate
import utils.tf_utils  as tf_utils


def prepare_data():
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = dp.preprocess_data(train_df, test_df)
    return (X_train, y_train), (X_test, y_test), sequence_length, vocab_size


def lstm_dropout_model_unprocessed_data():
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = dp.preprocess_data(train_df, test_df)
    # build and train model
    model = lstm_model.build_classifier_lstm_dropout(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    lstm_evaluate.evaluate(model, X_test, y_test)
    lstm_evaluate.render_plot(history)


def simple_model_simple_lemmitized_data():
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = dp.preprocess_data(train_df, test_df,
                                                                                           data_column="text")
    # build and train model
    model = lstm_model.build_classifier(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    lstm_evaluate.evaluate(model, X_test, y_test)
    lstm_evaluate.render_plot(history)


def main():
    # build and train model
    model = lstm_model.build_classifier_xx(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    lstm_evaluate.evaluate(model, X_test, y_test)
    lstm_evaluate.render_plot(history)


def export_plot_models():
    # TODO delete
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size = prepare_data()
    model_dropout = lstm_model.build_classifier_lstm_dropout(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_dropout)
    model_lstm2_dense2 = lstm_model.build_classifier_lstm2_dense2(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_lstm2_dense2)
    model_lstm3 = lstm_model.build_classifier_lstm3(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_lstm3)



if __name__ == '__main__':
    tf_utils.activate_compatibility_allow_growth()
    #lstm_dropout_model_unprocessed_data()
    export_plot_models()
