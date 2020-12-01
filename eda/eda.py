#!/usr/bin/env python
# coding: utf-8

# # EDA

# https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html                                        
# https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label                                               
# https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label

# In[1]:


#petras_path for plots 
path_plot = '/home/petra42/AIDA_Abschluss/aida_question_classification/plots/'


# ## Libaries

# In[2]:


import pandas as pd
import numpy as np

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


def process_question(row):
   return " ".join(row.split(" ")[1:])

train_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label", encoding = "ISO-8859-1", header=None)
train_df.columns = ["raw"]
train_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
train_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
train_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

train_df


# In[4]:


test_df = pd.read_table("https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label", encoding = "ISO-8859-1", header=None)
test_df.columns = ["raw"]
test_df['category'] = train_df.apply (lambda row: row["raw"].split(":")[0], axis=1)
test_df['subcategory'] = train_df.apply (lambda row: row["raw"].split(" ")[0].split(":")[1], axis=1)
test_df['question'] = train_df.apply (lambda row: process_question(row["raw"]), axis=1)

test_df


# In[5]:


train_df.head(5)

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

# In[6]:


# print test, train shape
print(f'---shapes---\ntrain:\t{train_df.shape}\ntest:\t{test_df.shape}')


# In[7]:


# number of row of columns
print(f'train_size:\t{train_df.size}\ntest_size:\t{test_df.size}') 


# In[8]:


# line occupancy    
print(f'---train---:\n {train_df.nunique()}\n')
print(f'---test---:\n {test_df.nunique()}')


# In[9]:


#which categories?
print(f"unique categories: {train_df['category'].unique()}")
#same categories in train and test?
train_df['category'].unique() == test_df['category'].unique()


# In[10]:


# load dictionary as dict
#which categories are not used?.
#which categories are used in train and test?

print(f"unique subcategories: {train_df['subcategory'].unique()}")




# In[11]:


# distribution categories (train)
dist_train_cat = train_df.groupby('category')['subcategory'].count()
print(dist_train_cat)
dist_train_cat.sort_values(ascending=False)

#visualisation 
dist_train_cat.plot.pie()
plt.title('Distribution categories (train)')
plt.savefig(path_plot + 'Distribution_categories_train.png')
plt.show()


# In[12]:


type(dist_train_cat)
dist_train_cat.values


# In[13]:


plt.pie(dist_train_cat,labels=dist_train_cat.index, autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.3) 
plt.title('Distribution categories (train)')
plt.savefig(path_plot + 'Distribution_categories_train%.png')
plt.show()


# In[47]:


# Pie chart, where the slices will show the max_category:
labels=dist_train_cat.index
sizes = dist_train_cat
explode = (0, 0, 0.1, 0, 0, 0)  # only "explode" the 3nd slice (max_category)

fig1, ax1 = plt.subplots()
ax1.pie(dist_train_cat, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution categories (train)')
plt.savefig(path_plot + 'Distribution_categories_train_slice.png')
plt.show()


# In[48]:


# distribution categories (test)
dist_test_cat = test_df.groupby('category')['subcategory'].count()
dist_test_cat.sort_values(ascending=False)
print(dist_test_cat)

plt.pie(dist_test_cat,labels=dist_test_cat.index, autopct='%1.0f%%', pctdistance=1.15, labeldistance=1.3) 
plt.title('Distribution categories (test)')
plt.savefig(path_plot + 'Distribution_categories_test%.png')
plt.show()


# In[49]:


# Pie chart, where the slices will show the max_categories:
labels=dist_test_cat.index
explode = (0, 0.1, 0.1, 0, 0, 0)  # only "explode" the slice (max_categories)

fig1, ax1 = plt.subplots()
ax1.pie(dist_test_cat, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution categories (test)')
plt.savefig(path_plot + 'Distribution_categories_test_slice.png')
plt.show()


# In[17]:


# distribution subcategories (train)
dist_train_sub = train_df.subcategory.groupby(train_df['category']).value_counts()
print(dist_train_sub)


# In[18]:


#first 20 subcategories
print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')


# In[50]:


fig= plt.figure(figsize=(12,6))
dist_train_sub.sort_values(ascending=False).plot.bar(label='train')
plt.ylabel('count question')
plt.title('Distribution subcategories (train)')
plt.savefig(path_plot + 'Distribution_subcategories_train.png')
plt.show()


# In[20]:


# distribution subcategories (test)
dist_test_sub = test_df.subcategory.groupby(test_df['category']).value_counts()
print(dist_test_sub)


# In[21]:


print(f'first 20: {dist_train_sub.sort_values(ascending=False)[:20]}')


# In[51]:


plt.figure(figsize=(12,6))
dist_test_sub.sort_values(ascending=False).plot.bar(label='test')
plt.ylabel('count question')
plt.title('Distribution subcategories (test)')
plt.savefig(path_plot + 'Distribution_subcategories_test.png')
plt.show()


# In[23]:


#plt.gca().set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})


# In[24]:


#test_df with only 38 subcategories compared to 47 in train_df
n = 14   #input for nlargest
train_df.subcategory.value_counts().nlargest(n).sum()

p = (train_df.subcategory.value_counts().nlargest(n).sum())/len(train_df)
print(f'The {n} largest subcategories account for a proportion of {p}')


# In[25]:


train_df.subcategory.value_counts().nlargest(n)

top_subcategories = list((train_df.subcategory.value_counts().nlargest(n)).index)
top_subcategories

train_df_top = train_df.loc[train_df['subcategory'].isin(top_subcategories)]
print(f'The {n} largest subcategories have {len(train_df_top)} questions.')


# In[26]:


train_df_top.sample(10)
train_df_top.subcategory.groupby(train_df['category']).value_counts()

#More than 80% of the data can be assigned to only 14 out of 47 subcategories in train_df

test_df.subcategory.value_counts().nlargest(39)


# In[52]:


#len text
plt.hist(train_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions (train)')
plt.savefig(path_plot + 'length_questions_train.png')
plt.show()


# In[53]:


#len text in test_df.question
plt.hist(test_df['question'].apply(lambda text: len(text.split())))
plt.xlabel('number of token')
plt.ylabel('number of question')
plt.title('Length of questions (test)')
plt.savefig(path_plot + 'length_questions_test.png')
plt.show()


# In[54]:


ax1 = plt.hist(train_df['question'].apply(lambda text: len(text.split())),label='train')
ax2 = plt.hist(test_df['question'].apply(lambda text: len(text.split())), label='test')
plt.legend()
plt.xlabel('number of token')
plt.ylabel('number of question per data set')
plt.title('Length of questions')
plt.savefig(path_plot + 'length_questions.png')
plt.show()


# In[30]:


#average length of question
al_train = round(train_df['question'].apply(lambda text: len(text.split())).mean())
al_test = round(test_df['question'].apply(lambda text: len(text.split())).mean())
print(f'The questions in the train data set have an average of {al_train} tokens, and {al_test} tokens in the test data set.')


# In[31]:


train_question_df = train_df['question'].str.split(" ", expand=True)
train_question_df 


# In[32]:


#percentage distribution of tokens (train)
for column in train_question_df.columns:
    nan_sum = train_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(train_question_df)),2))


# In[33]:


test_question_df = test_df['question'].str.split(" ", expand=True)
test_question_df


# In[34]:


#percentage distribution of tokens (test)
for column in test_question_df.columns:
    nan_sum = test_question_df[column].isnull().sum()
    print(column,': ',round(100.00-(nan_sum*100/len(test_question_df)),2))


# In[ ]:





# In[35]:


print(train_question_df.shape)
test_question_df.shape


# In[36]:


#distribution of most frequent first words in questions - top 15
question_top_words = train_question_df[0].value_counts().nlargest(15)
question_top_words


# In[37]:


question_words = list(train_question_df[0].value_counts().nlargest(15).index[0:6])
question_words  #top 7 from above


# In[38]:


#combination of words table with (sub)categories


# In[39]:


train_df_words = train_df.join(train_question_df)


# In[40]:


train_df_words_top = train_df_words.loc[train_df_words[0].isin(question_words)]


# In[41]:


#More than 90% of the question texts start with one of 7 (top) question words
train_df_words_top.shape


# In[42]:


#question words give indications for category
train_df_words_top[0].groupby(train_df_words_top['category']).value_counts()


# In[43]:


#train_df_words_top   only used for eda


# In[44]:


def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True


# In[45]:


checkIfDuplicates_1(list(train_df.question))


# In[46]:


checkIfDuplicates_1(list(test_df.question))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




