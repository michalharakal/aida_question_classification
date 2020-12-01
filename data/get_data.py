import pathlib
import pandas as pd
import os
import shutil
import requests


def download_file(url, dest_folder):
    local_filename = url.split('/')[-1] + ".txt"
    path = os.path.join(dest_folder, local_filename)
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


def local_data_exists(file_name):
    return os.path.isfile(file_name)


def local_data_missing(file_name):
    return not local_data_exists(file_name)


def check_local_data():
    """ Check existence of file in local directr and if they are missing, download them """
    if local_data_missing('train_5500.label.txt') or local_data_missing('train_5500.label.txt'):
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", "./")
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", "./")


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


def get_train_data(train_file='train_5500.label.txt'):
    """

    :param train_file:
    :return:
    """
    if local_data_missing(train_file):
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", "./data")

    return process_pd_input(train_file)


def get_test_data(test_file='TREC_10.label.txt'):
    """

    :param test_file:
    :return:
    """
    if local_data_missing(test_file):
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", "./data")

    return process_pd_input(test_file)


if __name__ == '__main__':
    check_local_data()
