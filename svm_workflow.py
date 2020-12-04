import pandas as pd
import matplotlib.pyplot as plt

import data.get_data as data
import utils.text_manipulation as txtm

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
    df_train['text'] = df_train.question.apply(txtm.clean_text)
    df_test['text'] = df_test.question.apply(txtm.clean_text)

    # apply stopword manipulation
    df_train['text'] = df_train.text.apply(txtm.stopword_text)
    df_test['text'] = df_test.text.apply(txtm.stopword_text)

    # apply lemmatizer
    df_train['text'] = df_train.text.apply(txtm.lem_text)
    df_test['text'] = df_test.text.apply(txtm.lem_text)

    return df_train, df_test


def preprocess_stpw_clean_lem(df_train, df_test):
    df_train = txtm.preprocess_dataframe(df_train)
    df_test = txtm.preprocess_dataframe(df_test)

    return df_train, df_test


def pipeline_results(pipeline, X_test, y_test, name='pipe'):
    """
    print pipeline results for given pipeline and X, y values

    :param name: unique to save class rep in report dir
    :param pipeline: fitted sklearn pipeline
    :param X_test: X_test values
    :param y_test: y_test values
    :return: None
    """

    y_pred_pipeline = pipeline.predict(X_test)

    print('accuracy_score: CountVectorized', accuracy_score(y_test, y_pred_pipeline))

    report = classification_report(y_test, y_pred_pipeline, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)
    print(df)
    # save df results to cvs for later report
    df.to_csv( './report/' + name + '.csv')

    # activate one or the other for of a confusion_matrix
    plot = True
    if plot:
        plot_confusion_matrix(pipeline, X_test, y_test, values_format="", cmap=plt.cm.Blues)
        plt.title(name)
        plt.savefig(f"plots/{name}.png")
        plt.show()
    else:
        val_confusion_matrix = confusion_matrix(y_test, y_pred_pipeline)
        print(f'Confusion Matrix: \n{val_confusion_matrix}')


def main():
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()

    # preprocess questions regex clean, stopwords, lemmatizer
    # df_train, df_test = preprocess(df_train, df_test)

    df_train, df_test = preprocess_stpw_clean_lem(df_train, df_test)

    # defining main test and train data of main categories
    # df col with text and cat/subcategory to be used
    #df_col = 'text_stopwords'
    df_col = 'question'
    # cat
    # category = 'subcategory'
    category = 'category'

    X_train = df_train[df_col]
    y_train = df_train[category]
    X_test = df_test[df_col]
    y_test = df_test[category]


    ##############################
    # prediction

    pipe_tf = Pipeline(steps=[
        ('data_vec', TfidfVectorizer()),
        ('model', svm.LinearSVC())  # C=1.0
    ])
    print('cross_val_score, TfidfVectorized :', cross_val_score(pipe_tf, X_train, y_train).mean())

    pipe_cv = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[])),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, CountVectorized  pipe_cv_no_stop_words: default ngram_range=(1, 1)',
          cross_val_score(pipe_cv, X_train, y_train).mean())

    pipe_cv_ng12 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(1, 2))),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_no_stop_words ngram_range=(1, 2) :',
          cross_val_score(pipe_cv_ng12, X_train, y_train).mean())

    pipe_cv_ng13 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(1, 3))),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_no_stop_words ngram_range=(1, 3) :',
          cross_val_score(pipe_cv_ng13, X_train, y_train).mean())

    pipe_cv_ng22 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(2, 2))),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_ng22 ngram_range=(2, 2) :',
          cross_val_score(pipe_cv_ng22, X_train, y_train).mean())

    pipe_cv_ng23 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(2, 3))),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_ng23 ngram_range=(2, 3) :',
          cross_val_score(pipe_cv_ng23, X_train, y_train).mean())

    pipe_cv_ng33 = Pipeline(steps=[
        ('data_cv', CountVectorizer(stop_words=[], ngram_range=(3, 3))),
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, pipe_cv_ng33 ngram_range=(3, 3) :',
          cross_val_score(pipe_cv_ng33, X_train, y_train).mean())

    ##############################
    # Measuring the performance
    # Testing the pipeline models using Count Vectorized and again with Tfi-df Vectorized.

    pipe_tf.fit(X_train, y_train)
    print('results TfidfVectorized pipeline fkt:')
    pipeline_results(pipe_tf, X_test, y_test, df_col + '_pipe_TfidfVect')

    pipe_cv.fit(X_train, y_train)
    print('results CountVectorized pipeline fkt:')
    pipeline_results(pipe_cv, X_test, y_test, df_col + '_pipe_cv')

    pipe_cv_ng12.fit(X_train, y_train)
    print('results pipe_cv_no_stop_words ngram_range=(1, 2) fkt:')
    pipeline_results(pipe_cv_ng12, X_test, y_test, df_col + '_pipe_cv_ng12')

    pipe_cv_ng13.fit(X_train, y_train)
    print('results pipe_cv_no_stop_words ngram_range=(1, 3) fkt:')
    pipeline_results(pipe_cv_ng13, X_test, y_test, df_col + '_pipe_cv_ng13')

    pipe_cv_ng22.fit(X_train, y_train)
    print('results pipe_cv_ng22 ngram_range=(2, 2) fkt:')
    pipeline_results(pipe_cv_ng22, X_test, y_test, df_col + '_pipe_cv_ng22')

    pipe_cv_ng23.fit(X_train, y_train)
    print('results pipe_cv_ng23 ngram_range=(2, 3) fkt:')
    pipeline_results(pipe_cv_ng23, X_test, y_test, df_col + '_pipe_cv_ng23')

    pipe_cv_ng33.fit(X_train, y_train)
    print('results pipe_cv_ng33 ngram_range=(3, 3) fkt:')
    pipeline_results(pipe_cv_ng33, X_test, y_test, df_col + '_pipe_cv_ng33')


if __name__ == '__main__':
    main()
