import sklearn
import math
import torch

import data.get_data as data
import utils.text_manipulation as txtm
from models.lstm import evaluate as transformer_evaluate

from transformers import AutoConfig, logging, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, \
    Trainer

# training params
lang_model = 'german-nlp-group/electra-base-german-uncased'
text_col_name = 'text'
label_col_name = 'coarse_label'
n_gpu = 1
max_seq_len = 100
label_list = []


# dataset to store tokenized text and labels
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# metrics util function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_list = sklearn.metrics.f1_score(labels, preds, labels=list(range(len(label_list))), average=None)
    recall_list = sklearn.metrics.recall_score(labels, preds, labels=list(range(len(label_list))), average=None)

    result_dict = {}
    result_dict.update({f'f1_{label_list[i]}': score for i, score in enumerate(f1_list)})
    result_dict.update({f'recall_{label_list[i]}': score for i, score in enumerate(recall_list)})
    result_dict['acc'] = sklearn.metrics.accuracy_score(labels, preds)
    result_dict['bac'] = sklearn.metrics.balanced_accuracy_score(labels, preds)
    result_dict['mcc'] = sklearn.metrics.matthews_corrcoef(labels, preds)
    result_dict['f1_macro'] = sklearn.metrics.f1_score(labels, preds, average='macro')

    return result_dict


def load_data():
    # read from file, parse string, create columns with question and categories
    test_df = data.get_test_data()
    train_df = data.get_train_data()
    return train_df, test_df


def encode_data(df, tokenizer):
    # preprocess
    train_df = txtm.preprocess_dataframe(df)

    data_text = train_df["question"].tolist()
    data_label = df["category"].tolist()

    # encode label
    global label_list
    label_list = df["category"].unique().tolist()
    data_label_encoded = [label_list.index(l) for l in data_label]
    assert len(data_label_encoded) == len(data_label)

    assert len(data_text) == len(data_label)

    data_text_encoded = tokenizer(
        data_text,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_token_type_ids=False,
    )

    labeled_dataset = LabeledDataset(data_text_encoded, data_label_encoded)

    print(f'LabeledDataset of len {len(labeled_dataset)} loaded. Source file: ')

    return labeled_dataset, data_label_encoded


# set hyperparameters
dropout = 0.3
learning_rate = 2.576e-05
num_train_epochs = 2  # 11 would be better
warmup_proportion = 0.2
batch_size_per_gpu = 16


def main():
    logging.set_verbosity_info()
    # load and create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    train_df, test_df = load_data()
    labeled_dataset_train, _ = encode_data(train_df, tokenizer)
    labeled_dataset_test, test_labels = encode_data(test_df, tokenizer)

    # calculate more parameters
    total_batch_size = batch_size_per_gpu * n_gpu
    num_train_samples = len(labeled_dataset_train)
    steps_per_epoch = math.ceil(num_train_samples / total_batch_size)
    steps_total = steps_per_epoch * num_train_epochs
    warmup_steps = int(warmup_proportion * steps_total)

    # model training config
    config = AutoConfig.from_pretrained(
        lang_model,
        num_labels=len(label_list),
    )
    config.summary_last_dropout = dropout

    # create and load model
    model = AutoModelForSequenceClassification.from_pretrained(
        lang_model,
        config=config,
    )

    # training config
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size_per_gpu,
        per_device_eval_batch_size=batch_size_per_gpu,
        learning_rate=learning_rate,
        gradient_accumulation_steps=1,
        warmup_steps=warmup_steps,

        output_dir='.',
        overwrite_output_dir=True,
        save_steps=0,
        logging_dir=None,
        logging_steps=0,
        evaluation_strategy='epoch',

        disable_tqdm=False,
    )

    # create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_dataset_train,
        eval_dataset=labeled_dataset_test,
        compute_metrics=compute_metrics,
    )

    # train and get results
    trainer.train()
    train_result = trainer.state.log_history[-2]

    predicted = trainer.predict(labeled_dataset_test)

    print(predicted[2])
    predicted_labels = list(predicted[1])
    transformer_evaluate.evaluate_transformer("transformer",  predicted_labels, test_labels)

    print(predicted)
    print(train_result)


if __name__ == '__main__':
    main()
    """
    lstm_dropout_model_unprocessed_data()
    lstm2_dense2_model_unprocessed_data()
    lstm3_model_unprocessed_data()
    """
