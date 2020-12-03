import numpy as np
import pandas as pd


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
