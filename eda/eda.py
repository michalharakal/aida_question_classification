#!/usr/bin/env python
# coding: utf-8

# EDA
#======================================================================================================================
#data files
path_class_def  = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html'
path_train_data = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
path_test_data  = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'

# In[1]:
path_plot = '/home/petra42/GIT/aida_question_classification/plots/'

# ## Libaries

import pandas as pd

#visualisation
import matplotlib.pyplot as plt

#
import data.get_data as data
import utils.text_manipulation as txtm


# In[3]:

#===== create data frame - train =====
df_train= pd.read_table(path_train_data, encoding = "ISO-8859-1", header=None)
df_train.columns = ["raw"]
df_train['category'] = df_train.apply (lambda row: row["raw"].split(":")[0], axis=1)
df_train['subcategory'] = df_train.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
df_train['question'] = df_train.apply (lambda row: data.process_question(row["raw"]), axis=1)

df_train.head()


# In[4]:
#===== create data frame - test =====
dt_test = pd.read_table(path_test_data, encoding = "ISO-8859-1", header=None)
df_test.columns = ["raw"]
df_test['category'] = df_train.apply (lambda row: row["raw"].split(":")[0], axis=1)
df_test['subcategory'] = df_train.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
df_test['question'] = df_train.apply (lambda row: process_question(row["raw"]), axis=1)

df_test.head()

#=====describe=====
df_train.describe()
df_test.describe()

# ======## QUESTION: ======
#     - shape, - size
#     - info: 
#         count: row, unique categories, subcategories
#     - witch catagories, subcategories, same in both dataframes
#     - distribution
#     - duplicates
#     - len questions (count token)

# print test, train shape
print(f'---shapes---\ntrain:\t{df_train.shape}\ntest:\t{df_test.shape}')

# number of row of columns
print(f'train_size:\t{df_train.size}\ntest_size:\t{df_test.size}')

# line occupancy    
print(f'---train---:\n {df_train.nunique()}\n')
print(f'---test---:\n {df_test.nunique()}')

#which categories?
print(f"unique categories: {df_train['category'].unique()}")
#same categories in train and test?
df_train['category'].unique() == df_test['category'].unique()


# In[10]:


# load dictionary as dict
#which categories are not used?.
#which categories are used in train and test?

print(f"unique subcategories: {df_train['subcategory'].unique()}")



# create corpus
def corpus_func(df, column):
    '''
    create a textcorpus from pd.series

    Parameters
    ----------
    text, string

    Return
    ------
    concatinated string with marker ##### as selector
    '''
    return "######".join(text for text in df)

def split_corpus_func(corpus):
    '''
    create a column from text corpus with marker '#####'
    as selector

    Parameters
    ----------
    text, string

    Return
    ------
    column in dafa frame
    '''
    return df

def preprocess_dataframe(df):
    '''
    create new columns in the data frame
    new_column: 'text' => cleaned stopwords (english)
                'text_clean' => regex, lowercase
                'text_lemma' => lemmetized
    param: df
    returns: df with new columns
    '''

    corpus = corpus_func(df['question'])
    text_corpus = txtm.stopword_text(corpus)
    df['text'] = split_corpus_func(corpus)
    clean_corpus = clean_text(text_corpus)
    df['text_clean'] = split_corpus_func(clean_corpus)
    lemma = txtm.lem_text(clean_corpus)
    df['text_lemma'] = split_corpus_func(clean_corpus)
    return df




def main():
    # loading data and working with pd df
    df_train = data.get_train_data()
    df_test = data.get_test_data()

if __name__ == '__main__':

    preprocess_dataframe(df_train)
    # main()
    
