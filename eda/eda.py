# -*- coding: utf-8 -*-
"""Kopie von Group_A_AN_EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SJI3LUb1Tu66tqd6mA1SUcQpVq0uPd1u

"""

# In[1]:
# EDA

# https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html 
# https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label
# https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label

#%%md
##Libaries

# In[2]:
import pandas as pd
import numpy as np

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#%%md  
##Load & Explore

# In[4]:
def process_question(row):
   return " ".join(row.split(" ")[1:])

train_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", encoding = "ISO-8859-1", header=None)
train_df.columns = ["raw"]
train_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
train_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
train_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

train_df

# In[5]:

test_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", encoding = "ISO-8859-1", header=None)
test_df.columns = ["raw"]
test_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
test_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
test_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

test_df

# In[6]:
    
train_df.head(5)
train_df.columns  

#describe
train_df.describe()
test_df.describe()

#%%md
# 
# ## question:
#     - shape 
#     - size
#     - info: 
#         count: row, unique categories, subcategories
#     - witch catagories, subcategories, same in both dataframes
#     - distribution
#     - len questions (count token)
#    

# In[7]: 
    
# print test, train shape
print(f'shapes:\ntrain:\t{train_df.shape}\ntest:\t{test_df.shape}')

# In[8]:
   
# number of row of columns
print(f'train_size:\t{train_df.size}\ntest_size:\t{test_df.size}')    
    
# In[9]: 
    
# line occupancy    
print(f'---train---: {train_df.nunique()}\n')
print(f'---test---: {test_df.nunique()}')


#which categories?
print(f"unique catagories: {train_df['category'].unique()}")
#same categories in train and test?
train_df['category'].unique() == test_df['category'].unique()
  
# In[10]:     
    
# distribution categories (train)
dist_train_cat = train_df.groupby('category')['subcategory'].count()
print(dist_train_cat)
dist_train_cat.sort_values(ascending=False)

#visualisation 
dist_train_cat.plot.pie()
plt.title('catagories train data')
plt.show()

# In[11]:  

# distribution categories (test)
dist_test_cat = test_df.groupby('category')['subcategory'].count()
dist_test_cat.sort_values(ascending=False)
print(dist_test_cat)

dist_test_cat.plot.pie()
plt.title('catagories test data')
plt.show()

# In[12]:  

# distribution subcategories (train)
dist_train_sub = train_df.subcategory.groupby(train_df['category']).value_counts()
print(dist_train_sub,'\n')
print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')

fig= plt.figure(figsize=(12,6))
dist_train_sub.sort_values(ascending=False).plot.bar(label='train')
plt.ylabel('count question')
plt.title('Distribution subcategories, train')
plt.show()

# In[13]:  
    
# distribution subcategories (test)

dist_test_sub = test_df.subcategory.groupby(test_df['category']).value_counts()
print(dist_test_sub, '\n')
print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')

fig= plt.figure(figsize=(12,6))
dist_test_sub.sort_values(ascending=False).plot.bar(label='test')
plt.ylabel('count question')
plt.title('Distribution subcategories, test')
plt.show()

# In[14]:  

#test_df with only 38 subcategories compared to 47 in train_df
n = 14   #input for nlargest
train_df.subcategory.value_counts().nlargest(n).sum()

p = (train_df.subcategory.value_counts().nlargest(n).sum())/len(train_df)
print(f'The {n} largest subcategories account for a proportion of {p}')

# In[15]:  

train_df.subcategory.value_counts().nlargest(n)

top_subcategories = list((train_df.subcategory.value_counts().nlargest(n)).index)
top_subcategories

train_df_top = train_df.loc[train_df['subcategory'].isin(top_subcategories)]
len(train_df_top)

# In[16]: 

train_df_top.sample(10)
train_df_top.subcategory.groupby(train_df['category']).value_counts()

#More than 80% of the data can be assigned to only 14 out of 47 subcategories in train_df

test_df.subcategory.value_counts().nlargest(39)

# In[17]: 

#len text
plt.hist(train_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions, traindata')
plt.show()
  

# In[18]: 

#len text in test_df.question
plt.hist(test_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions, testdata')
plt.show()

 
# In[19]:  

#len text (all)    
ax1 = plt.hist(train_df['question'].apply(lambda text: len(text.split())),label='train')
ax2 = plt.hist(test_df['question'].apply(lambda text: len(text.split())), label='test')
plt.legend()
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions')
plt.show()  

 
# In[20]:  
    
#average length of question
al_train = round(train_df['question'].apply(lambda text: len(text.split())).mean())
al_test = round(test_df['question'].apply(lambda text: len(text.split())).mean())
print(f'The questions in the train data set have an average of {al_train} tokens, \
and {al_test} tokens in the test data set.')

# In[21]:

train_question_df = train_df['question'].str.split(" ", expand=True)
train_question_df 
    
# In[22]:  
    
#percentage distribution of tokens (train)
for column in train_question_df.columns:
    nan_sum = train_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(train_question_df)),2))   

# In[23]:


test_question_df = test_df['question'].str.split(" ", expand=True)
test_question_df   

# In[24]:


#percentage distribution of tokens (test)
for column in test_question_df.columns:
    nan_sum = test_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(test_question_df)),2)) 


if __name__ == '__main__' :
    main()

