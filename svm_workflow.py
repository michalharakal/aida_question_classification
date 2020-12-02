import pandas as pd

import data.get_data as data
import utils.text_manipulation as text

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, classification_report


def preprocess(df_train, df_test):
    """
    cleaning text file / question column of DF with regex, of stopwords and apply lemmatizer

    :param df_train: train DF
    :param df_test: test DF
    :return:
        df_train, df_test Data Frames
    """

    # apply regex textcleaning
    df_train['text'] = df_train.question.apply(text.clean_text)
    df_test['text'] = df_test.question.apply(text.clean_text)

    # apply stopword manipulation
    df_train['text'] = df_train.text.apply(text.stopword_text)
    df_test['text'] = df_test.text.apply(text.stopword_text)

    # apply lemmatizer
    df_train['text'] = df_train.text.apply(text.lem_text)
    df_test['text'] = df_test.text.apply(text.lem_text)

    return df_train, df_test


def evaluate_pipeline(pipeline, X_test, y_test):
    """
    print pipeline results for given pipeline and X, y values

    :param pipeline: fitted sklearn pipeline
    :param X_test: X_test values
    :param y_test: y_test values
    :return: None
    """

    y_pred_pipeline = pipeline.predict(X_test)

    print('accuracy_score: CountVectorized', accuracy_score(y_test, y_pred_pipeline))

    report = classification_report(y_test, y_pred_pipeline)
    print(report)

    val_confusion_matrix = confusion_matrix(y_test, y_pred_pipeline)
    print(f'Confusion Matrix: \n{val_confusion_matrix}')


def evaluate():
    """

    Returns
    -------

    """
    pass


def main():
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()

    # preprocess questions regex clean, stopwords, lemmatizer
    df_train, df_test = preprocess(df_train, df_test)

    # defining main test and train data of main categories
    # X_train = df_train['text'] # question - data regex, stopwords, lem
    X_train = df_train['question'] # question - data not cleaned
    y_train = df_train['category']  # using main category

    # X_test = df_test['text']  # question - data not cleaned
    X_test = df_test['question'] # question - data not cleaned
    y_test = df_test['category']  # using main category

    # Test - defining main test and train data of main sub categories
    # X_train = df_train.text
    # y_train = df_train['subcategory']  # using main category
    # X_test = df_test.text
    # y_test = df_test['subcategory']  # using main category

    # create CountVectorizer to validate in df the use of it.
    # later not used directly only applied in Pipeline
    # ngram_range=(1, 2),
    count_vectorizer = CountVectorizer(stop_words=[], ngram_range=(1, 3))
    bag_of_words = count_vectorizer.fit_transform(X_train)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

    # print(type(bag_of_words))
    print('bag_of_words shape', bag_of_words.shape)

    # create vectorizer out of words of questions
    # later not used directly only applied in Pipeline
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

    # Show the Model as a pandas DataFrame
    feature_names = tfidf_vectorizer.get_feature_names()
    df_tfidf_vectorizer = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # print(type(tfidf_matrix))
    print(tfidf_matrix.shape)


    # prediction

    pipe_tf = Pipeline(steps=[
        ('data_vec', TfidfVectorizer()),
        # ('model', LogisticRegressionCV())
        ('model', svm.LinearSVC())  # C=10.0
    ])
    print('cross_val_score, TfidfVectorized :', cross_val_score(pipe_tf, X_train, y_train).mean())

    pipe_cv = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[])),
        # ('model', LogisticRegressionCV())
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, CountVectorized  pipe_cv_no_stop_words:', cross_val_score(pipe_cv, X_train, y_train).mean())

    pipe_cv_ng12 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(1, 2))),
        # ('model', LogisticRegressionCV())
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_no_stop_words ngram_range=(1, 2) :', cross_val_score(pipe_cv_ng12, X_train, y_train).mean())

    pipe_cv_ng13 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(1, 3))),
        # ('model', LogisticRegressionCV())
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_no_stop_words ngram_range=(1, 3) :',
          cross_val_score(pipe_cv_ng13, X_train, y_train).mean())

    ##############################
    # Measuring the performance
    # Testing the pipeline models using Count Vectorized and again with Tfi-df Vectorized.

    pipe_tf.fit(X_train, y_train)
    print('results TfidfVectorized pipeline fkt:')
    evaluate_pipeline(pipe_tf, X_test, y_test)

    pipe_cv.fit(X_train, y_train)
    print('results CountVectorized pipeline fkt:')
    evaluate_pipeline(pipe_cv, X_test, y_test)

    pipe_cv_ng12.fit(X_train, y_train)
    print('results pipe_cv_no_stop_words ngram_range=(1, 2) fkt:')
    evaluate_pipeline(pipe_cv_ng12, X_test, y_test)

    pipe_cv_ng13.fit(X_train, y_train)
    print('results pipe_cv_no_stop_words ngram_range=(1, 3) fkt:')
    evaluate_pipeline(pipe_cv_ng13, X_test, y_test)

if __name__ == '__main__':
    main()
