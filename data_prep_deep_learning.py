# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:26:00 2021

@author: Admin
"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


"""
Importing and Analyzing the Dataset

"""

filename =r'C:\Users\Admin\Documents\Cours\MATH80629\Project\Analysis\dataset.txt'
dataset = pd.read_csv(filename,sep=",")
print(dataset.head(5))

data_worry = dataset.loc[:, ['worry','text_long']]

#see an example of the dataset
data_worry.dtypes      
print(data_worry.head(5))
data_worry['text_long'][3]

#see the distribution of target variable
data_worry['worry'].value_counts()
plt.hist(data_worry['worry'], bins=9)


#Data Preprocessing functions

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

#Data Preprocessing 
X = []
sentences = list(data_worry['text_long'])
for sen in sentences:
    X.append(preprocess_text(sen))

    
y=data_worry['worry']

#split train test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,stratify=y)


"""
Preparing the Embedding Layer

"""

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#data_worry.text_long.str.len().mode()

#GloVe embeddings
embeddings_dictionary = dict()
glove_file = open(r'C:\Users\Admin\Documents\Cours\MATH80629\Project\Analysis\glove.6B.100d.txt', encoding="utf8")


for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


#GloVe embeddings matrix with glove dimension 100
embedding_matrix = np.zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

"""

Simple Neural Network

"""

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.compile(optimizer="Adam", loss="mse", metrics=["mae",'mape'])

print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


