import utils.text_manipulation as txtm
import pandas as pd
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.preprocessing.text import Tokenizer


def common_tokenizer(df_train, df_test, column="question"):
    data = pd.concat([df_train, df_test])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data[column].values)
    return tokenizer


def tokenize(tokenizer, df_text, column="question"):
    sequences = tokenizer.texts_to_sequences(df_text[column].values)
    iterator = iter(tokenizer.word_index.items())
    for i in range(10):
        print(next(iterator))

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    print('\nSize of vocabulary: ', vocab_size)
    return sequences, vocab_size


def create_sequences(preprocessed_train, preprocessed_test, column="question"):
    tokenizer = common_tokenizer(preprocessed_train, preprocessed_test, column)
    # train
    tokenized_train, vocab_size_train = tokenize(tokenizer, preprocessed_train)
    train_X = sequence.pad_sequences(tokenized_train, value=0.)
    max_len_train = max(set([len(x) for x in tokenized_train]))
    # test
    tokenized_test, vocab_size_test = tokenize(tokenizer, preprocessed_test)
    max_len_test = max(set([len(x) for x in tokenized_test]))

    # take the same value for padding (max length)
    max_len = max(max_len_train, max_len_test)
    test_X = sequence.pad_sequences(tokenized_test, maxlen=train_X.shape[1], value=0.)

    return train_X, test_X, train_X.shape[1], vocab_size_train, tokenizer


def encode_classes(train_df, test_df, category_col="category"):
    y_train = pd.get_dummies(train_df[category_col])
    y_test = pd.get_dummies(test_df[category_col])

    return y_train.values, y_test.values


def preprocess_data(train_df, test_df, data_column="question", classes_column="category"):
    train_df = txtm.preprocess_dataframe(train_df)
    test_df = txtm.preprocess_dataframe(test_df)
    sequenced_train, sequenced_test, sequence_length, vocab_size_train, tokenizer = \
        create_sequences(train_df, test_df, data_column)
    encoded_train, encoded_test = encode_classes(train_df, test_df, classes_column)
    return (sequenced_train, encoded_train), (
    sequenced_test, encoded_test), sequence_length, vocab_size_train, tokenizer
