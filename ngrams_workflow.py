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
    sorted_items = sorted_items[-topn :]


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
    count_vectorizer = CountVectorizer(stop_words=None, ngram_range=ngram_range )
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

def main():
    """
    Testing some function in file
    """
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()
    categories = df_train.category.unique()



    print(categories)
    X_train_desc = df_train[df_train.category == 'DESC' ]['question']
    X_train = df_train['question']

    count_vectorizer, df_bag_of_words, bag_of_words = ng_CountVectorizer(X_train_desc, (2, 3))


    # sort the vocab
    # sorted_items = sort_vocab(bag_of_words[0].tocoo())
    # Get feature names (words/n-grams). It is sorted by position in sparse matrix
    # feature_names = count_vectorizer.get_feature_names()
    # n_grams = extract_topn_from_vector(feature_names, sorted_items, 10)

    # doch anders.....

    # Get full vocab names
    vocabulary = count_vectorizer.vocabulary_

    max_words = dict((word, df_bag_of_words[word].sum()) for word in vocabulary)
    smax_words = sorted(max_words, key=max_words.get, reverse=True)[:10]

    print(smax_words)
    for word in smax_words:
        print( word, ':', max_words[word])


    print()

if __name__ == '__main__':
    main()