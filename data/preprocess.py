from pandas import CategoricalDtype

import utils.text_manipulation as text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def preprocess_text(df_train_raw, df_test_raw):
    df_train = df_train_raw.copy()
    df_test = df_test_raw.copy()

    # apply regex text cleaning
    df_train['text'] = df_train.question.apply(text.clean_text)
    df_test['text'] = df_test.question.apply(text.clean_text)

    # apply stopword manipulation
    df_train['text'] = df_train.text.apply(text.stopword_text)
    df_test['text'] = df_test.text.apply(text.stopword_text)

    # apply lemmatizer
    df_train['text'] = df_train.text.apply(text.lem_text)
    df_test['text'] = df_test.text.apply(text.lem_text)

    return df_train, df_test


def vectorize_words(df_train, df_test):
    count_vectorizer = CountVectorizer()
    bag_of_words = count_vectorizer.fit_transform(df_train)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
    return df_bag_of_words


def vectorize_words_tfid(df_train, df_test):
    # create vectorizer out of words of questions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_train)

    # Show the Model as a pandas DataFrame
    feature_names = tfidf_vectorizer.get_feature_names()
    df_tfidf_vectorizer = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # print(type(tfidf_matrix))
    return df_tfidf_vectorizer.shape


def tokenize(df_text, column="question"):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_text[column])
    sequences = tokenizer.texts_to_sequences(df_text[column])
    print('\nSequences is of type:      ', type(sequences))
    print('\nFirst sequence is of type: ', type(sequences[0]))
    print('\nFirst sequence looks like this: ')
    print(sequences[0])

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    print('\nSize of vocabulary: ', vocab_size)
    return sequences


def create_pads(tokenized):
    return sequence.pad_sequences(tokenized, padding='post', truncating="post")


def create_sequences(preprocessed_train, preprocessed_test):
    # train
    tokenized_train = tokenize(preprocessed_train)
    sequenced_train = create_pads(tokenized_train)

    # test
    tokenized_test = tokenize(preprocessed_test)
    sequenced_test = create_pads(tokenized_test)

    return sequenced_train, sequenced_test, sequenced_train.shape[1]


def encode_classes(train_df, test_df, category_col="category"):

    y_train = train_df.copy()
    train_categories = train_df[category_col].unique()
    cat_type = CategoricalDtype(categories=train_categories, ordered=True)
    y_train[category_col + "_coded"] = y_train[category_col].astype(cat_type).cat.codes

    y_test = test_df.copy()
    test_categories = y_test[category_col].unique()
    cat_type = CategoricalDtype(categories=test_categories, ordered=True)
    y_test[category_col + "_coded"] = y_test[category_col].astype(cat_type).cat.codes

    return y_train[category_col + "_coded"], y_test[category_col + "_coded"]


def preprocess_data(train_df, test_df):
    preprocessed_train, preprocessed_test = preprocess_text(train_df, test_df)
    sequenced_train, sequenced_test, sequence_length = create_sequences(preprocessed_train, preprocessed_test)
    encoded_train, encoded_test = encode_classes(train_df, test_df)
    return (sequenced_train, encoded_train), (sequenced_test, encoded_test), sequence_length


def train_test_split(train_df, test_df):
    """
    Make train test split

    Returns
    -------
    (X_train, y_train), (X_test, y_test):
        pandas.Serie with splitred test and train data

    """
    X_train = train_df["text"]
    X_test = test_df["category"]

    y_train = train_df["text"]
    y_test = train_df["category"]

    return (X_train, y_train), (X_test, y_test)
