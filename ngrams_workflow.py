import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import data.get_data as data



def ng_CountVectorizer(X_train, ngram_range):

    # create CountVectorizer to validate in df the use of it.

    # later not used directly only applied in Pipeline
    # ngram_range=(1, 2),
    count_vectorizer = CountVectorizer(stop_words=[], ngram_range=(1, 3))
    bag_of_words = count_vectorizer.fit_transform(X_train)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

    # print(type(bag_of_words))
    # print('bag_of_words shape', bag_of_words.shape)
    return df_bag_of_words, bag_of_words


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


if __name__ == '__main__':
    main()