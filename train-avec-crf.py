#!/usr/bin/env python
# coding: utf-8

# In[69]:


## Importing all required packages
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras.layers import TimeDistributed, Dropout
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import LSTM, Embedding, Dense
import unicodedata


# In[70]:


## Loading data 
train_data = pd.read_csv("new_train_data.csv")
test_data = pd.read_csv("new_test_data.csv")


# In[71]:


train_data


# In[72]:


test_data


# In[73]:


##-------------------------------------- Data Analysis ----------------------------------- ##

print("Training data summarization\n:",train_data.nunique())


# In[74]:


## Getting the list of words
words = list(set(train_data["Word"].append(test_data["Word"]).values))


# In[75]:


words


# In[76]:


## Creating the vocabulary
## Converting into ascii form
words = [unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore') for w in words]
n_words = len(words)


# In[77]:


words


# In[78]:


n_words


# In[79]:


## Creating the list of tags
tags = list(set(train_data["tag"].values))
n_tags = len(tags)


# In[80]:


tags


# In[81]:


n_tags


# In[82]:


## Converting into index and map with words and tag in order to refer for future.
word_idx = {word:index for index, word in enumerate(words)}
tag_idx = {tag:index for index, tag in enumerate(tags)}


# In[83]:


word_idx


# In[84]:


tag_idx


# In[85]:


##------------------------------------ Preparing the dataset --------------------------------------------#
word_tag_func = lambda s: [(word,tag) for word, tag in zip(s["Word"].values, s["tag"].values)]
grouped_word_tag = train_data.groupby("Sent_ID").apply(word_tag_func)
sentences = [s for s in grouped_word_tag]


# In[86]:


grouped_word_tag


# In[87]:


sentences


# In[88]:


word_tag_func = lambda s: [word for word in s["Word"].values]
grouped_word = test_data.groupby("Sent_ID").apply(word_tag_func )
test_sentences = [s for s in grouped_word]


# In[89]:


grouped_word


# In[90]:


test_sentences


# In[91]:


##------------------------------- Preparing data for modelling ----------------------------##
X_train = [[word_idx[unicodedata.normalize('NFKD', str(w[0])).
encode('ascii','ignore')] for w in s]for s in sentences]


# In[92]:


X_train


# In[93]:


## Preparing input training data
X_train = pad_sequences(sequences= X_train, maxlen=180, padding='post')


# In[94]:


X_train


# In[95]:


print(len(X_train))


# In[96]:


print(len(Y_train))


# In[97]:


X_test = [[word_idx[unicodedata.normalize('NFKD', str(w)).
encode('ascii','ignore')] for w in s] for s in test_sentences]


# In[98]:


X_test


# In[99]:


## Preparing input test data
X_test = pad_sequences(sequences= X_test, maxlen= 180, padding='post')


# In[100]:


X_test


# In[ ]:





# In[101]:


## Preparing output training data




# In[102]:


y_train = [[tag_idx[w[1]] for w in s] for s in sentences]


# In[103]:


y_train = pad_sequences(sequences=y_train, maxlen=180, padding= 'post', value= tag_idx["O"])


# In[104]:


y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]


# In[105]:


Y_train = np.array(y_train)


# In[106]:


print(len(Y_train))


# In[107]:


y_train


# In[108]:


print(len(y_train))


# In[109]:


Y_train


# In[110]:


##------------------------------------------------ Model Creation -----------------------------------------------##

input = Input(shape=(180,))


# In[111]:


input


# In[112]:


model = Embedding(input_dim= n_words, output_dim=180, input_length=180)(input)


# In[113]:


model


# In[114]:


model = Dropout(0.1)(model)


# In[115]:


model


# In[116]:


model = LSTM(units=150, return_sequences=True, recurrent_dropout=0.1)(model)


# In[117]:


model


# In[118]:


model = TimeDistributed(Dense(n_tags, activation="relu"))(model)


# In[119]:


model


# In[120]:


crf_model = CRF(n_tags+1)


# In[121]:


crf_model


# In[122]:


output = crf_model(model)  # output


# In[123]:


output


# In[124]:


model = Model(input, output)


# In[125]:


model


# In[126]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[127]:


print(model.summary)


# In[128]:


fitted_model = model.fit(X_train, y_train, batch_size= 48, epochs= 5, verbose=1)


# In[ ]:




