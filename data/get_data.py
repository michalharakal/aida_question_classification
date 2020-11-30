
import re
import pathlib
import pandas as pd

def process_question(row):
   return " ".join(row.split(" ")[1:])

def process_pd_input(file_name):
    path = pathlib.Path.cwd().joinpath('data', file_name)
    df = pd.read_table(path, encoding="ISO-8859-1", header=None)
    df.columns = ["raw"]
    df['category'] = df.apply(lambda row: row["raw"].split(":")[0], axis=1)
    df['subcategory'] = df.apply(lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
    df['question'] = df.apply(lambda row: process_question(row["raw"]), axis=1)

    return df


def get_train_data(train_file = 'train_5500.label.txt'):
    """

    :param train_file:
    :return:
    """
    return process_pd_input(train_file)

def get_test_data(test_file = 'TREC_10.label.txt'):
    """

    :param test_file:
    :return:
    """
    return process_pd_input(test_file)
