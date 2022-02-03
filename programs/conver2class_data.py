#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:40:35 2022

@author: user
"""

import numpy as np 

######################## read data ################################
headlines = np.load('data/REN-20k_headline_abstract_data.npy')
labels = np.load('data/REN-20k_headline_abstract_labels.npy') #anger, fear, joy, sad
