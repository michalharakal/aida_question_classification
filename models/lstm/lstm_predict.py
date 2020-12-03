import numpy as np
import pandas as pd
from keras.preprocessing import sequence


def print_predictions(model, tokenizer, test_X, test_df, category_column="category"):
    """
    :param model: trained model
    :param tokenizer: instance of tokenizer used for training
    :param test_X: DataFrame with test data
    :param category_column:  column with categories
    """
    predictions = model.predict(test_X)

    reverse_dictionary = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))

    test_categories = pd.get_dummies(test_df[category_column]).columns

    for i in range(len(test_X[:10])):
        frase = ''
        for j in test_X[i]:
            if j != 0:
                frase += reverse_dictionary[j] + ' '
        print('Frase ->', frase)
        print('    Label category->', test_df[category_column][i])
        print('    Label predict->', test_categories[np.argmax(predictions[i])])
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def predict_lstm(model, tokenizer, train_df, question):
    tokenized_question = tokenizer.texts_to_sequences([question])
    max_question_length = train_df["question"].str.len().max()
    padded_question = sequence.pad_sequences(tokenized_question, maxlen=max_question_length, value=0.)
    prediction = model.predict(padded_question)
    y_pred_class_index = np.argmax(prediction, axis=1)
    test_categories = pd.get_dummies(train_df["category"]).columns
    predicted_class_label = np.array([test_categories[index] for index in y_pred_class_index])
    return predicted_class_label
