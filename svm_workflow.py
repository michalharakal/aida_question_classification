import data.get_data as data
import utils.text_manipulation as text

import pandas as pd


def train():
    """

    Returns
    -------

    """
    pass


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

    # apply regex textcleaning
    df_train['text'] = df_train.question.apply(text.clean_text)
    df_test['text'] = df_test.question.apply(text.clean_text)

    # apply stopword manipulation
    df_train['text'] = df_train.text.apply(text.stopword_text)
    df_test['text'] = df_test.text.apply(text.stopword_text)

    # apply lemmatizer
    df_train['text'] = df_train.text.apply(text.lem_text)
    df_test['text'] = df_test.text.apply(text.lem_text)

    X_train = df_train.text
    y_train = df_train['category']

    X_test = df_test.text
    y_test = df_test['category']

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    count_vectorizer = CountVectorizer()
    bag_of_words = count_vectorizer.fit_transform(X_train)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

    #print(type(bag_of_words))
    print(bag_of_words.shape)

    # create vectorizer out of words of questions
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

    # Show the Model as a pandas DataFrame
    feature_names = tfidf_vectorizer.get_feature_names()
    df_tfidf_vectorizer = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # print(type(tfidf_matrix))
    print(tfidf_matrix.shape)

    # prediction
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn import svm

    pipe_tf = Pipeline(steps=[
        ('data_vec', TfidfVectorizer()),
        #('model', LogisticRegressionCV())
        ('model', svm.LinearSVC()) # C=10.0
    ])
    print('cross_val_score, TfidfVectorizer :', cross_val_score(pipe_tf, X_train, y_train).mean())

    pipe_cv = Pipeline(steps=[
        ('data_cv', CountVectorizer()),
        #('model', LogisticRegressionCV())
        ('model', svm.LinearSVC())
    ])
    print('cross_val_score, CountVectorizer :', cross_val_score(pipe_cv, X_train, y_train).mean())

    # Measuring the performance
    # Testing the model using Count Vectoriser and again with Tfi-df Vectoriser.

    from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, classification_report

    pipe_tf.fit(X_train, y_train)
    # plot_confusion_matrix(pipe_tf, X_test, y_test)

    pipeline_preds = pipe_tf.predict(X_test)
    print('accuracy_score: TfidfVectorizer', accuracy_score(y_test, pipeline_preds))

    report = classification_report(y_test, pipeline_preds)
    print(report)

    val_confusion_matrix = confusion_matrix(y_test, pipeline_preds)
    print(f'Confusion Matrix: \n{val_confusion_matrix}')

    print()

    pipe_cv.fit(X_train, y_train)
    pipeline_preds = pipe_cv.predict(X_test)
    print('accuracy_score: CountVectorizer', accuracy_score(y_test, pipeline_preds))

    report = classification_report(y_test, pipeline_preds)
    print(report)

    val_confusion_matrix = confusion_matrix(y_test, pipeline_preds)
    print(f'Confusion Matrix: \n{val_confusion_matrix}')

    #train()


    #evaluate()


if __name__ == '__main__':
    main()
