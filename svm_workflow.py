import data.get_data as data
import utils.text_manipulation as text


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
    df_train['text'] = df_train.question.apply(text.stopword_text)
    df_test['text'] = df_test.question.apply(text.stopword_text)

    # apply lemmatizer
    df_train['text'] = df_train.question.apply(text.lem_text)
    df_test['text'] = df_test.question.apply(text.lem_text)

    #train()


    #evaluate()


if __name__ == '__main__':
    main()
