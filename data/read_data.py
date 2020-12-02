import os

import pandas as pd

dir_name = 'data/question-classif-data'
train_filename = 'train_5500.label'
test_filename = 'TREC_10.label'


def check_files_exist():
    filenames = [train_filename, test_filename]
    num_files = len(filenames)
    for i in range(len(filenames)):
        file_exists = os.path.isfile(os.path.join(dir_name, filenames[i]))
        assert file_exists
    print('Requested files exist')


def read_data_into_pandas(filename):
    col_names = ['Category', 'Subcategory', 'Question']
    data = pd.DataFrame(columns=col_names)
    with open(filename, 'r', encoding='latin-1') as f:
        count = 1;
        for row in f:
            # Quebra a linha em duas no primeiro espaço (category:subcategory and question)
            row_str = row.split(" ", 1)
            # Separa a categoria da subcategoria
            row_category = row_str[0].split(":")
            data.loc[len(data)] = [row_category[0], row_category[1], row_str[1].lower()]
            count += 1
    return data


def read_all_with_concat():
    data_train = read_data_into_pandas(os.path.join(dir_name, train_filename))
    data_test = read_data_into_pandas(os.path.join(dir_name, test_filename))

    return data_train, data_test
