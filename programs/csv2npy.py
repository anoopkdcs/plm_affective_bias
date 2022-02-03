#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 10:41:58 2022

@author: Anoop
Data set csv 2 npy 
data = tweets 
labels = [0:anger, 1:fear, 2:joy, 3:sadness]
"""

import csv
import numpy as np 

#### train data ####
train_file = open('data/semEval/train.csv')
train_csvreader = csv.reader(train_file)
train_header = next(train_csvreader)
train_rows = []

for i in train_csvreader:
        train_rows.append(i)
train_rows = np.array(train_rows)   

#train_data = np.reshape(train_rows[:,0],(len(train_rows[:,0]),1))
train_data  = train_rows[:,0]
train_label = np.reshape(train_rows[:,1], (len(train_rows[:,1]),1))


np.save('data/semEval/train_data.npy',train_data)
np.save('data/semEval/train_labels.npy', train_label) 


#### validation data ####
val_file = open('data/semEval/val.csv')
val_csvreader = csv.reader(val_file)
val_header = next(val_csvreader)
val_rows = []

for j in val_csvreader:
    val_rows.append(j)
val_rows = np.array(val_rows)

#val_data = np.reshape(val_rows[:,0],(len(val_rows[:,0]),1))
val_data = val_rows[:,0]
val_label = np.reshape(val_rows[:,1], (len(val_rows[:,1]),1))

np.save('data/semEval/val_data.npy',val_data)
np.save('data/semEval/val_labels.npy', val_label) 


#### test data ####
test_file = open('data/semEval/test.csv')
test_csvreader = csv.reader(test_file)
test_header = next(test_csvreader)
test_rows = []

for k in test_csvreader:
        test_rows.append(k)
test_rows = np.array(test_rows)

#test_data = np.reshape(test_rows[:,0],(len(test_rows[:,0]),1))
test_data = test_rows[:,0]
test_label = np.reshape(test_rows[:,1], (len(test_rows[:,1]),1))

# Save news and coresponding Labels 
np.save('data/semEval/test_data.npy',test_data)
np.save('data/semEval/test_labels.npy', test_label) 


