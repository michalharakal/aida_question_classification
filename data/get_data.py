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
    """
        read files from data dir, process down to needed columns

        :param file_name: training file filename
        :return: splited data frame
    """

    path = pathlib.Path.cwd().joinpath('data', file_name)
    if path.exists():
        path = pathlib.Path.cwd().joinpath('data', file_name)
    else:
        path = pathlib.Path.cwd().joinpath(file_name)
    # print(path)

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
    if local_data_missing(train_file):
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", "./data")

    return process_pd_input(train_file)


def get_test_data(test_file='TREC_10.label.txt'):
    """
    get testing file

    :param test_file: test file filename
        default TREC_10.label.txt
    :return: test data frame
    """
    if local_data_missing(test_file):
        download_file("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", "./data")

    return process_pd_input(test_file)


def main():
    """
    Testing some function in file
    """
    df = process_pd_input('TREC_10.label.txt')
    # print(df['cat_combined'])


if __name__ == '__main__':
    check_local_data()
    # main()
