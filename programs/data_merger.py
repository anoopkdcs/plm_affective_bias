#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:37:01 2022

@author: Data Merger
"""
import numpy as np 

train_data = np.load('data/semEval/train_data.npy')
train_labels = np.load('data/semEval/train_labels.npy')

val_data = np.load('data/semEval/val_data.npy')
val_labels = np.load('data/semEval/val_labels.npy')

train_data_fused = np.concatenate((train_data,val_data),axis=0)
train_labels_fused = np.concatenate((train_labels,val_labels),axis=0)

np.save('data/semEval/train_data_fused.npy',train_data_fused)
np.save('data/semEval/train_labels_fused.npy', train_labels_fused) 