#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('cd', '')


# In[2]:


get_ipython().system('pwd')


# In[3]:


get_ipython().run_line_magic('cd', 'Downloads/ML_Projects/Sentiment Analysis')


# In[4]:


get_ipython().run_line_magic('cd', 'sentiment-analysis-on-movie-reviews')


# In[5]:


get_ipython().system('pwd')


# In[6]:


get_ipython().run_line_magic('ls', '')


# In[7]:


data = pd.read_table('train.tsv')


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data.info()


# In[11]:


import string


# In[12]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def preprocessor(s):
    s = s.split()
    for i in range(0,len(s)):
        k = s.pop(0)
        if k not in string.punctuation and k not in stopwords:
                s.append(lemmatizer.lemmatize(k).lower())    
    return s


# In[13]:


X = data['Phrase']
y = data['Sentiment']


# In[14]:


print(preprocessor(X[81]))


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer = preprocessor)
X_train_count = count_vect.fit_transform(X_train)
X_train_count.shape


# In[17]:


from sklearn.feature_extraction.text import TfidfTransformer
tfid_transformer = TfidfTransformer()
X_train_tfid = tfid_transformer.fit_transform(X_train_count)
X_train_tfid.shape


# In[18]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier().fit(X_train_tfid, y_train)


# In[19]:


X_test = count_vect.transform(X_test)
X_test_tfidf = tfid_transformer.transform(X_test)
predictions = clf.predict(X_test_tfidf)


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))


# In[21]:


test_data = pd.read_table('test.tsv')


# In[22]:


test_data.head()


# In[23]:


phrase_id = test_data['PhraseId'].values
test_data = count_vect.transform(test_data['Phrase'])
test_data_tfidf = tfid_transformer.transform(test_data)
test_predictions = clf.predict(test_data_tfidf)


# In[24]:


test_predictions.shape


# In[25]:


prediction = pd.DataFrame({'PhraseId':phrase_id, 'Sentiment' : test_predictions})


# In[26]:


prediction.head()


# In[27]:


csv_file = 'Movie Review Prediction.csv'
prediction.to_csv(csv_file, index = False)
print('Saved File : ' + csv_file)

