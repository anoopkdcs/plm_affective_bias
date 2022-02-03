#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:11:15 2022

@author: EEC2Test conversion
"""
import csv
import numpy as np 

#### Read EEC  ####
eec_file = open('data/EEC/eec_emotiononly.csv')
eec_csvreader = csv.reader(eec_file)
eec_header = next(eec_csvreader)
eec_rows = []

for i in eec_csvreader:
        eec_rows.append(i)
eec_rows = np.array(eec_rows)  

eec_data =  eec_rows[:,1]
eec_label_original = np.reshape(eec_rows[:,6], (len(eec_rows[:,1]),1))
eec_label = eec_label_original
eec_label = np.where(eec_label=='anger',0,eec_label)
eec_label = np.where(eec_label=='fear',1,eec_label)
eec_label = np.where(eec_label=='joy',2,eec_label)
eec_label = np.where(eec_label=='sadness',3,eec_label)

np.save('data/EEC/eec_data.npy',eec_data)
np.save('data/EEC/eec_label_original.npy', eec_label_original) 
np.save('data/EEC/eec_label.npy', eec_label)
