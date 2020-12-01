import re
import pathlib
import pandas as pd


def process_question(row):
    return " ".join(row.split(" ")[1:])


def process_pd_input(file_name):
    """
        read files from data dir, process down to needed columns

        :param file_name: training file filename
        :return: splited data frame
    """
    path = pathlib.Path.cwd().joinpath('data', file_name)
    df = pd.read_table(path, encoding="ISO-8859-1", header=None)
    df.columns = ["raw"]
    df['cat_combined'] = df.apply(lambda row: row["raw"].split(" ")[0], axis=1)
    df['category'] = df.apply(lambda row: row["raw"].split(":")[0], axis=1)
    df['subcategory'] = df.apply(lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
    df['question'] = df.apply(lambda row: process_question(row["raw"]), axis=1)

    return df


def get_train_data(train_file='train_5500.label.txt'):
    """
    get training file

    :param train_file: training file filename
        default train_5500.label.txt
    :return: train data frame
    """
    return process_pd_input(train_file)


def get_test_data(test_file='TREC_10.label.txt'):
    """
    get testing file

    :param test_file: test file filename
        default TREC_10.label.txt
    :return: test data frame
    """
    return process_pd_input(test_file)


def main():
    """
    Testing some function in file
    """


if __name__ == '__main__':
    main()