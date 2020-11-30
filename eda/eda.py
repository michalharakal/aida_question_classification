#!/usr/bin/env python
# coding: utf-8

# # EDA

# https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html                                        
# https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label                                               
# https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label

# ## Libaries

# In[1]:


import pandas as pd
import numpy as np

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def process_question(row):
   return " ".join(row.split(" ")[1:])

train_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", encoding = "ISO-8859-1", header=None)
train_df.columns = ["raw"]
train_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
train_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
train_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

train_df


# In[3]:


def process_question(row):
   return " ".join(row.split(" ")[1:])

test_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", encoding = "ISO-8859-1", header=None)
test_df.columns = ["raw"]
test_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
test_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
test_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

test_df


# In[4]:


train_df.head(5)
train_df.columns  

#describe
train_df.describe()
test_df.describe()


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
# 
#  

# In[5]:


# print test, train shape
print(f'---shapes---\ntrain:\t{train_df.shape}\ntest:\t{test_df.shape}')


# In[6]:


# number of row of columns
print(f'train_size:\t{train_df.size}\ntest_size:\t{test_df.size}') 


# In[7]:


# line occupancy    
print(f'---train---:\n {train_df.nunique()}\n')
print(f'---test---:\n {test_df.nunique()}')


# In[8]:


#which categories?
print(f"unique catagories: {train_df['category'].unique()}")
#same categories in train and test?
train_df['category'].unique() == test_df['category'].unique()


# In[9]:


# load dictionary as dict
#which categories are not used?.
#which categories are used in train and test?

print(f"unique subcatagories: {train_df['subcategory'].unique()}")




# In[10]:


# distribution categories (train)
dist_train_cat = train_df.groupby('category')['subcategory'].count()
print(dist_train_cat)
dist_train_cat.sort_values(ascending=False)

#visualisation 
dist_train_cat.plot.pie()
plt.title('Distribution catagories (train)')
plt.show()


# In[11]:


type(dist_train_cat)
dist_train_cat.values


# In[12]:


plt.pie(dist_train_cat,labels=dist_train_cat.index, autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.3) 
plt.title('Distribution catagories (train)')
plt.show()


# In[13]:


# Pie chart, where the slices will show the max_categorie:
labels=dist_train_cat.index
sizes = dist_train_cat
explode = (0, 0, 0.1, 0, 0, 0)  # only "explode" the 3nd slice (max_categorie)

fig1, ax1 = plt.subplots()
ax1.pie(dist_train_cat, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution catagories (train)')
plt.show()


# In[14]:


# distribution categories (test)
dist_test_cat = test_df.groupby('category')['subcategory'].count()
dist_test_cat.sort_values(ascending=False)
print(dist_test_cat)

plt.pie(dist_test_cat,labels=dist_test_cat.index, autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.3) 
plt.title('Distribution catagories (test)')
plt.show()


# In[15]:


# Pie chart, where the slices will show the max_categorie:
labels=dist_test_cat.index
sizes = dist_test_cat
explode = (0, 0.1, 0.1, 0, 0, 0)  # only "explode" the slice (max_categorie)

fig1, ax1 = plt.subplots()
ax1.pie(dist_test_cat, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution catagories (test)')
plt.show()


# In[16]:


# distribution subcategories (train)
dist_train_sub = train_df.subcategory.groupby(train_df['category']).value_counts()
print(dist_train_sub)


# In[17]:


#first 20 subcategories
print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')


# In[18]:


fig= plt.figure(figsize=(12,6))
dist_train_sub.sort_values(ascending=False).plot.bar(label='train')
plt.ylabel('count question')
plt.title('Distribution subcategories (train)')
plt.show()


# In[19]:


# distribution subcategories (test)
dist_test_sub = test_df.subcategory.groupby(test_df['category']).value_counts()
print(dist_test_sub)


# In[20]:


print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')


# In[21]:


fig= plt.figure(figsize=(12,6))
dist_test_sub.sort_values(ascending=False).plot.bar(label='test')
plt.ylabel('count question')
plt.title('Distribution subcategories (test)')
plt.show()


# In[22]:


#plt.gca().set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})


# In[23]:


#test_df with only 38 subcategories compared to 47 in train_df
n = 14   #input for nlargest
train_df.subcategory.value_counts().nlargest(n).sum()

p = (train_df.subcategory.value_counts().nlargest(n).sum())/len(train_df)
print(f'The {n} largest subcategories account for a proportion of {p}')


# In[24]:


train_df.subcategory.value_counts().nlargest(n)

top_subcategories = list((train_df.subcategory.value_counts().nlargest(n)).index)
top_subcategories

train_df_top = train_df.loc[train_df['subcategory'].isin(top_subcategories)]
print(f'The {n} largest subcategories have {len(train_df_top)} questions.')


# In[25]:


train_df_top.sample(10)
train_df_top.subcategory.groupby(train_df['category']).value_counts()

#More than 80% of the data can be assigned to only 14 out of 47 subcategories in train_df

test_df.subcategory.value_counts().nlargest(39)


# In[26]:


#len text
plt.hist(train_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions (train)')
plt.show()


# In[27]:


#len text in test_df.question
plt.hist(test_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions (test)');


# In[28]:


ax1 = plt.hist(train_df['question'].apply(lambda text: len(text.split())),label='train')
ax2 = plt.hist(test_df['question'].apply(lambda text: len(text.split())), label='test')
plt.legend()
plt.xlabel('number of token')
plt.ylabel('number of question per data set')
plt.title('Length of questions, testdata')
plt.show()


# In[29]:


#average length of question
al_train = round(train_df['question'].apply(lambda text: len(text.split())).mean())
al_test = round(test_df['question'].apply(lambda text: len(text.split())).mean())
print(f'The questions in the train data set have an average of {al_train} tokens, and {al_test} tokens in the test data set.')


# In[30]:


train_question_df = train_df['question'].str.split(" ", expand=True)
train_question_df 


# In[31]:


#percentage distribution of tokens (train)
for column in train_question_df.columns:
    nan_sum = train_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(train_question_df)),2))


# In[32]:


test_question_df = test_df['question'].str.split(" ", expand=True)
test_question_df


# In[33]:


#percentage distribution of tokens (test)
for column in test_question_df.columns:
    nan_sum = test_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(test_question_df)),2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True


# In[35]:


checkIfDuplicates_1(list(train_df.question))


# In[36]:


checkIfDuplicates_1(list(test_df.question))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




