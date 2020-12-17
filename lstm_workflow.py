import data.get_data as data
import data.preprocess as dp
from models.lstm import lstm_model as lstm_model
from models.lstm import evaluate as lstm_evaluate
import models.lstm.lstm_predict as lstm_predict
import utils.report_utils as rutils


def evaluate_model(X_test, history, model, y_test):
    test_df = data.get_test_data()
    report_df = lstm_evaluate.evaluate_lstm(model, X_test, y_test, test_df)
    rutils.df_to_styled_img(report_df, model.name)
    lstm_evaluate.training_accuracy_plot(model, history)
    lstm_evaluate.test_confusion_matrix(model, X_test, y_test, test_df)


def prepare_data(data_column="question", classes_column="category"):
    # get data
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    # preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = dp.preprocess_data(train_df, test_df,
                                                                                                      data_column,
                                                                                                      classes_column)
    return (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer


def lstm_dropout_model_unprocessed_data():
    # get and preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = prepare_data()
    # build and train model
    model = lstm_model.build_classifier_lstm_dropout(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    evaluate_model(X_test, history, model, y_test)
    test_df = data.get_test_data()
    lstm_predict.print_predictions(model, tokenizer, X_test, test_df)


def lstm3_model_unprocessed_data():
    # get and preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = prepare_data()
    # build and train model
    model = lstm_model.build_classifier_lstm3(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    evaluate_model(X_test, history, model, y_test)
    test_df = data.get_test_data()
    lstm_predict.print_predictions(model, tokenizer, X_test, test_df)


def lstm2_dense2_model_unprocessed_data():
    # get and preprocess data
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = prepare_data()
    # build and train model
    model = lstm_model.build_classifier_lstm2_dense2(vocab_size, sequence_length, y_train.shape[1])
    history = lstm_model.train(model, X_train, y_train)
    # evaluate
    evaluate_model(X_test, history, model, y_test)
    test_df = data.get_test_data()
    lstm_predict.print_predictions(model, tokenizer, X_test, test_df)


def evaluate_plot_models():
    (X_train, y_train), (X_test, y_test), sequence_length, vocab_size, tokenizer = prepare_data()
    model_dropout = lstm_model.build_classifier_lstm_dropout(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_dropout)
    model_lstm2_dense2 = lstm_model.build_classifier_lstm2_dense2(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_lstm2_dense2)
    model_lstm3 = lstm_model.build_classifier_lstm3(vocab_size, sequence_length, y_train.shape[1])
    lstm_model.plot_model_summary(model_lstm3)


if __name__ == '__main__':
    # tf_utils.activate_compatibility_allow_growth()
    lstm_dropout_model_unprocessed_data()
    lstm2_dense2_model_unprocessed_data()
    lstm3_model_unprocessed_data()
