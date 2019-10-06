#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:49:56 2019

@author: vipss
"""

#import required libraries
import pandas as pd#for data
import numpy as np#for preprocessing and training
import re#for text processing
import nltk#for text processing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler#scaling
from sklearn.model_selection import train_test_split
from keras.models import Sequential#for training
from keras.layers import Dense, LSTM, SpatialDropout1D, Embedding#for training
from keras.callbacks import EarlyStopping#for training

#download the stopwords, for text processing
nltk.download('stopwords')

#global variables required in future
REPLACE_BY_SPACE_RE=re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
#max num of words
MAX_NB_WORDS=40000
#maximum words in each question
MAX_SEQUENCE_LENGTH=250
#embedding
EMBEDDING_DIM=100

#functions
def clean_text(text):
    """
        text:a string
        return: clean string
    """
    for i in range(len(text)):
        text[i]=text[i].lower()
        text[i]=REPLACE_BY_SPACE_RE.sub(' ', text[i])#replace bad ones by extra spaces.
        text[i]=BAD_SYMBOLS_RE.sub('', text[i])#replace numbers and symbols by nothing
        text[i] = ' '.join(word for word in text[i].split() if word not in STOPWORDS)#remove stopwords
    return text

def scl(x):
    """
        x: training data
        return: x(scaled training data)
                scl(Standard Scalar)
    """
    final_sc=StandardScaler()
    x_final=final_sc.fit_transform(x)
    return x_final, final_sc

#read data line-by-line
company_x = [line.rstrip('\n') for line in open('apple-computers.txt')]
apple_x = [line.rstrip('\n') for line in open('apple-fruit.txt')]

#store it in a single variable
x=[]
x.extend(company_x)
x.extend(apple_x)

#store the target
y=[]
for i in range(len(company_x)):
    y.append(1)
for i in range(len(apple_x)):
    y.append(0)

#text processing
x=clean_text(x)#cleaning the text
#tokenizing the sentences given
tokenizer=Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(x)
x=tokenizer.texts_to_sequences(x)
x=pad_sequences(x, MAX_SEQUENCE_LENGTH)#padding all sequences for ease in training

#building the final model using neural networks and lstms
final_model=Sequential()
final_model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
final_model.add(SpatialDropout1D(0.2))
final_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
final_model.add(Dense(output_dim=1, activation="sigmoid"))
final_model.compile(metrics=['accuracy'], loss="binary_crossentropy", optimizer="adam")
testing=final_model.fit(x, y, epochs=10, batch_size=10, validation_split=0.01, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

#input
n=int(input())
inp=[]
for i in range(n):
    inp.append(input())
inp=clean_text(inp)
inp=tokenizer.texts_to_sequences(inp)
inp=pad_sequences(inp, MAX_SEQUENCE_LENGTH)
y_pred=final_model.predict(inp)
for i in y_pred:
    if i>=0.9:
        print('computer-company')
    else:
        print('fruit')