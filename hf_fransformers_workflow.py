import pandas as pd
import sklearn
import math
import numpy as np
import random
import torch

import data.get_data as data
import data.preprocess as dp

from transformers import AutoConfig, logging, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import datasets

# training params
lang_model = 'german-nlp-group/electra-base-german-uncased'
train_data_file = './models/transformers/germeval18/train.tsv'
test_data_file = './models/transformers/germeval18/test.tsv'
label_list = ["OTHER", "OFFENSE"]
text_col_name = 'text'
label_col_name = 'coarse_label'
n_gpu = 1


def encode_data(df, tokenizer):
    # read from file, parse string, create columns with question and categories
    test_df = data.get_test_data()
    train_df = data.get_train_data()

    # preprocess
    train_df = data.preprocess_data_only(train_df)
    test_df = data.preprocess_data_only(test_df)


    """

    data_text = df[text_col_name].tolist()
    data_label = df[label_col_name].tolist()
        assert len(data_text) == len(data_label)

        # encode label
        data_label_encoded = [label_list.index(l) for l in data_label]
        assert len(data_label_encoded) == len(data_label)

        data_text_encoded = tokenizer(
            data_text,
            truncation=True,
            padding='max_length',
            max_length=max_seq_len,
            return_token_type_ids=False,
        )

        labeled_dataset = LabeledDataset(data_text_encoded, data_label_encoded)

        print(f'LabeledDataset of len {len(labeled_dataset)} loaded. Source file: {filename}')

        return labeled_dataset, data_label_encoded
        """


def main():

    logging.set_verbosity_info()
    # load and create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    load_data()



if __name__ == '__main__':
    main()
    """
    lstm_dropout_model_unprocessed_data()
    lstm2_dense2_model_unprocessed_data()
    lstm3_model_unprocessed_data()
    """
