import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import data.get_data as data


def sort_vocab(vocab_matrix):
    tuples = zip(vocab_matrix.col, vocab_matrix.data)
    tup_sorted = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    return tup_sorted


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """return n-gram counts in asc order of counts"""

    # use only topn items from vector
    sorted_items = sorted_items[-topn:]

    results = []

    for idx, count in sorted_items:
        # get the ngram name
        n_gram = feature_names[idx]
        results.append((n_gram, count))

    return results


def ng_CountVectorizer(X_train, ngram_range=(1, 3)):
    """
    create CountVectorizer to validate in df the use of it.
    Parameters
    ----------
    X_train
    ngram_range

    Returns
    -------

    """
    print('ngram_range=', ngram_range)
    count_vectorizer = CountVectorizer(stop_words=None, ngram_range=ngram_range)
    bag_of_words = count_vectorizer.fit_transform(X_train)

    # returns the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

    return count_vectorizer, df_bag_of_words, bag_of_words


def ng_TfidfVectorizer(X_train, ngram_range):
    # create vectorizer out of words of questions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

    # Show the Model as a pandas DataFrame
    feature_names = tfidf_vectorizer.get_feature_names()
    df_tfidf_vectorizer = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # print(type(tfidf_matrix))
    # print(tfidf_matrix.shape)
    return df_tfidf_vectorizer, tfidf_matrix


def get_category_texts(df, category, df_col):
    """
    abstract category text request to test with different texts
    Parameters
    ----------
    df: data frame
    category: category to be returned
    df_col: df column text to process

    Returns
    df df col to process
    -------

    """
    return df[df.category == category][df_col]


def get_category_ngrams(x_train, ngram_range=(1, 1), topn=10):
    count_vectorizer, df_bag_of_words, bag_of_words = ng_CountVectorizer(x_train, ngram_range)

    # Get full vocab names
    vocabulary = count_vectorizer.vocabulary_

    max_words = dict((word, df_bag_of_words[word].sum()) for word in vocabulary)
    smax_words = sorted(max_words, key=max_words.get, reverse=True)[:topn]

    print('-----')
    print(smax_words)
    for word in smax_words:
        print(word, ':', max_words[word])


def main():
    """
    Testing some function in file
    """
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()
    categories = df_train.category.unique()
    print(categories)

    # X_train = df_train['question'] # all full text
    # X_train_desc = df_train[df_train.category == 'DESC']['question']
    # category = 'DESC'

    # default setting for ngram viz
    df_col = 'question'
    ngram_range = (2, 3)
    topn = 10

    for category in categories:
        print()
        print('--', category, '---')
        x_train = get_category_texts(df_train, category, df_col)
        get_category_ngrams(x_train, ngram_range, topn)

    print()


if __name__ == '__main__':
    main()
