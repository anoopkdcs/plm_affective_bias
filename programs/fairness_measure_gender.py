#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:29:25 2022

@author: Individual fairness : 
"""

import csv
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
'''
fig = plt.figure(figsize = (10, 5))
plt.plot(range(len(male_anger_matrix)),male_anger_matrix, label = "male anger", linestyle="-")
plt.plot(female_anger_matrix, label = "female anger", linestyle="-")
plt.legend()
plt.show()
'''



def fairness_measures(male_matrix, female_matrix, emotion, emotion_label):
    
    male_emotion_indices = np.where(male_matrix[:,6] == emotion)
    male_emotion_matrix = male_matrix[male_emotion_indices[0],9]

    female_emotion_indices = np.where(female_matrix[:,6] == emotion)
    female_emotion_matrix = female_matrix[female_emotion_indices[0],9]  
    
    #Individual fairness pllot 
    male_emotion_indices = np.where(male_matrix[:,6] == emotion)
    male_emotion_matrix = male_matrix[male_emotion_indices[0],9]
    male_emotion_list = male_emotion_matrix.tolist()
    male_emotion_list = [float(i) for i in male_emotion_list] 

    female_emotion_indices = np.where(female_matrix[:,6] == emotion)
    female_emotion_matrix = female_matrix[female_emotion_indices[0],9]  
    female_emotion_list = female_emotion_matrix.tolist()
    female_emotion_list = [float(i) for i in female_emotion_list] 

    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(len(male_emotion_list)),male_emotion_list, label = "male "+ str(emotion), color ='r') 
    plt.plot(range(len(female_emotion_list)), female_emotion_list, label = "female "+str(emotion), color ='b')
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend()
    plt.show()
    
    #Statistical significance over emotions
    value, pvalue = ttest_ind(male_emotion_list, female_emotion_list, equal_var=False)
    print("significance test values for male and female with emotion "+ str(emotion), value, pvalue)
    
    # Average delta  for Anger
    delta_emotion = np.absolute(np.reshape(np.float128(male_matrix[male_emotion_indices[0],9]), (1050,1)) - np.reshape(np.float128(female_matrix[female_emotion_indices[0],9]), (1050,1)))
    avg_delta_emotion = np.average(delta_emotion)
    print("average delta "+str(emotion)+ ": ",avg_delta_emotion)

    # Demographic Parity
    male_emotion_prediction_indices = np.array(np.where(male_matrix[:,8]== emotion_label))
    male_emotion_prediction_count = male_emotion_prediction_indices.shape[1]    
    prob_emotion_male = male_emotion_prediction_count/male_matrix.shape[0]
    print("Male "+str(emotion)+ "= ",prob_emotion_male)

    female_emotion_prediction_indices = np.array(np.where(female_matrix[:,8]==emotion_label))
    female_emotion_prediction_count = female_emotion_prediction_indices.shape[1]
    prob_emotion_female = female_emotion_prediction_count/female_matrix.shape[0]
    print("Female "+str(emotion)+ "= ",prob_emotion_female)  
    
    
    return

#### Read EEC  ####
eec_prediction_file = open('data/eec_predictions/eec_emotion_only_bert.csv')
eec_prediction_csvreader = csv.reader(eec_prediction_file)
eec_prediction_header = next(eec_prediction_csvreader)
eec_prediction_rows = []

for i in eec_prediction_csvreader:
        eec_prediction_rows.append(i)
        
eec_prediction_rows = np.array(eec_prediction_rows)  
#eec_prediction_rows = eec_prediction_rows[0:1200,:]

#eec_prediction_rows_df = pd.DataFrame(eec_prediction_rows, columns = ['ID','Sentence','Template','Person','Gender','Race'
                                                   #   ,'Emotion', 'Emotion_word', 'Predictions', 'Prediction_Intensity' ])

male_indices = np.where(eec_prediction_rows[:,4] == 'male')
male_matrix = eec_prediction_rows[male_indices[0],:] 

female_indices = np.where(eec_prediction_rows[:,4] == 'female')
female_matrix = eec_prediction_rows[female_indices[0],:] 

male_predictions_only= male_matrix[:,9]
female_predictions_only= female_matrix[:,9]

male_list = male_predictions_only.tolist()
male_list = [float(i) for i in male_list]

female_list = female_predictions_only.tolist()
female_list = [float(i) for i in female_list]

#Student t-test for RMSE
#https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
value, pvalue = ttest_ind(male_list, female_list, equal_var=False)
print("significance test values for male and female:", value, pvalue)


fairness_measures(male_matrix, female_matrix, 'anger', '0')
fairness_measures(male_matrix, female_matrix, 'fear', '1')
fairness_measures(male_matrix, female_matrix, 'joy', '2')
fairness_measures(male_matrix, female_matrix, 'sadness', '3')


