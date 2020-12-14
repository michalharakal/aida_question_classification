import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data.get_data as data
import utils.text_manipulation as txtm
import models.spacy_module as spacy_mod

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin  # for definition of custom transformers

from sklearn.pipeline import Pipeline, FeatureUnion
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

    # testing
    df_train['spacy'] = df_train.question.apply(spacy_mod.process_question)
    df_test['spacy'] = df_test.question.apply(spacy_mod.process_question)

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
    df.to_csv('./report/svm_' + name + '.csv')

    # activate one or the other for of a confusion_matrix
    plot = False
    if plot:
        plot_confusion_matrix(pipeline, X_test, y_test, values_format="", cmap=plt.cm.Blues)
        plt.title(name)
        plt.savefig(f"plots/{name}.png")
        plt.show()
    else:
        val_confusion_matrix = confusion_matrix(y_test, y_pred_pipeline)
        print(f'Confusion Matrix: \n{val_confusion_matrix}')


class TextExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def transform(self, data):
        return np.asarray(data[self.column]).astype(str)

    def fit(self, *_):
        return self


class SpacyTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        spacyed = pd.DataFrame(X, columns=['spacy'])
        spacyed = spacyed.spacy.str.split('|', expand=True)
        return spacyed

    def fit(self, X, y=None, **fit_params):
        return self


def main():
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()

    # preprocess questions regex clean, stopwords, lemmatizer
    # df_train, df_test = preprocess(df_train, df_test)

    df_train, df_test = preprocess_stpw_clean_lem(df_train, df_test)

    # defining main test and train data of main categories
    # df col with text and cat/subcategory to be used
    df_col_tdf = df_col = 'question'  # 'text_stopwords'

    X_train_cols = df_train[[df_col, 'spacy']]
    X_test_cols = df_test[[df_col, 'spacy']]

    # cat
    #category = 'subcategory'
    category = 'category'

    X_train = df_train[df_col]
    y_train = df_train[category]
    X_test = df_test[df_col]
    y_test = df_test[category]


    # simple applied to df does not work in pipeline
    # df_spacy_train = pd.get_dummies(df_train.spacy.str.split('|', expand=True))
    # df_spacy_test = pd.get_dummies(df_test.spacy.str.split('|', expand=True))

    # so we have to build pipelines for the features

    spacy_transformer = Pipeline([
        ('desc_extractor', TextExtractor('spacy')),
        ('SpacyTransformer', SpacyTransformer()),
        ('one_hot', OneHotEncoder()),
    ])

    desc_df_col_featurizer = Pipeline([
        ('desc_extractor', TextExtractor(df_col_tdf)),
        ('CountVectorizer', CountVectorizer(stop_words=[], ngram_range=(1, 2)))
        #, ('tf_idf', TfidfTransformer())


    ])

    combined_features = FeatureUnion([("spacy_transformer", spacy_transformer), ("data_vec", desc_df_col_featurizer)])

    ##############################
    # prediction

    pipe_tf = Pipeline(steps=[
        ('data_vec', desc_df_col_featurizer),
        ('model', svm.LinearSVC())  # C=1.0
    ])

    pipe_tf_fu = Pipeline(steps=[
        ('combined_features_train', combined_features),
        ('model', svm.LinearSVC())  # C=1.0
    ])

    """
    
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
    
    """
    ##############################
    # Measuring the performance
    # Testing the pipeline models using Count Vectorized and again with Tfi-df Vectorized.

    pipe_tf.fit(X_train_cols, y_train)
    print('results TfidfVectorized pipeline fkt:')
    pipeline_results(pipe_tf, X_test_cols, y_test, df_col + '_pipe_TfidfVect')

    pipe_tf_fu.fit(X_train_cols, y_train)
    print('results TfidfVectorized pipeline fkt:')
    pipeline_results(pipe_tf_fu, X_test_cols, y_test, df_col + '_pipe_TfidfVect')

    """
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
    """


if __name__ == '__main__':
    main()
