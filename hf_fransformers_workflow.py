import pandas as pd
import sklearn
import math
import numpy as np
import random
import torch

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


def main():

    logging.set_verbosity_info()
    # load and create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lang_model)


if __name__ == '__main__':
    main()
    """
    lstm_dropout_model_unprocessed_data()
    lstm2_dense2_model_unprocessed_data()
    lstm3_model_unprocessed_data()
    """
