#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 11:49:53 2022

@author: user
"""
import numpy as np 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from collections import Counter

######################## read data ################################
train_data = np.load('data/semEval/train_data.npy')
val_data = np.load('data/semEval/val_data.npy')
test_data = np.load('data/semEval/test_data.npy')

### Sent Conter
def sent_counter(data):
    sent = np.zeros((0,1))
    for i in range(len(data)):
        doc = data[i]
        number_of_sentences = sent_tokenize(doc)
        sent = np.append(sent,len(number_of_sentences))
        
    avg_sent = np.sum(sent)/len(data)
    return avg_sent

### word Conter (avg words and unique words)
def word_count(data):
    Tokens = []
    totalLength = 0
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(data)):
        tempTokens = data[i].lower() #converting to lower case
        tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;"))
        tempTokens = tokenizer.tokenize(tempTokens)
        Tokens.append(tempTokens)
        totalLength = totalLength + len(Tokens[i])
    AvgWordperDocument = totalLength/len(data)
    
    #Unique number of words 
    totalWordlist = []
    for j in range(len(Tokens)):
        totalWordlist.extend(Tokens[j])   
    wrd_counter = Counter(totalWordlist)
        
    return AvgWordperDocument,len(wrd_counter)





#Sentence Counting
train_sent = sent_counter(train_data)
print(train_sent) #1.6627710504083357

val_sent = sent_counter(val_data)
print(val_sent) #1.6509562841530054

test_sent = sent_counter(test_data)
print(test_sent) #1.5457650273224044


# average word per document
train_avg_word, train_uniqu_words = word_count(train_data)
print(train_avg_word) #16.204308645451984
print(train_uniqu_words) #16604

val_avg_word, val_uniqu_words = word_count(val_data)
print(val_avg_word) #16.372267759562842
print(val_uniqu_words) #4406

test_avg_word, test_unique_words = word_count(test_data)
print(test_avg_word) #15.712431693989071
print(test_unique_words) #6461
