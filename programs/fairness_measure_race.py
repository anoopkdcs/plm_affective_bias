#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 23:06:11 2022

@author: user
"""

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

'''
fig = plt.figure(figsize = (10, 5))
plt.plot(range(len(male_anger_matrix)),male_anger_matrix, label = "male anger", linestyle="-")
plt.plot(female_anger_matrix, label = "female anger", linestyle="-")
plt.legend()
plt.show()
'''



def fairness_measures(african_matrix, european_matrix, emotion, emotion_label):
    
    african_emotion_indices = np.where(african_matrix[:,6] == emotion)
    african_emotion_matrix = african_matrix[african_emotion_indices[0],9]


    european_emotion_indices = np.where(european_matrix[:,6] == emotion)
    european_emotion_matrix = european_matrix[european_emotion_indices[0],9]  

    #Individual fairness pllot 
    african_emotion_indices = np.where(african_matrix[:,6] == emotion)
    african_emotion_matrix = african_matrix[african_emotion_indices[0],9]
    african_emotion_list = african_emotion_matrix.tolist()
    african_emotion_list = [float(i) for i in african_emotion_list] 

    european_emotion_indices = np.where(european_matrix[:,6] == emotion)
    european_emotion_matrix = european_matrix[european_emotion_indices[0],9]  
    european_emotion_list = european_emotion_matrix.tolist()
    european_emotion_list = [float(i) for i in european_emotion_list] 

    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(len(african_emotion_list)),african_emotion_list, label = "african "+ str(emotion), color ='r') 
    plt.plot(range(len(european_emotion_list)), european_emotion_list, label = "european "+str(emotion), color ='b')
    plt.xlabel('Input Sentences', fontsize = 12)
    plt.ylabel('Predicted Emotion Intensity', fontsize = 12)
    plt.legend()
    plt.show()
    
    #Statistical significance over emotions
    value, pvalue = ttest_ind(african_emotion_list, european_emotion_list, equal_var=False)
    print("significance test values for african and european with emotion "+ str(emotion), value, pvalue)
    
    # Average delta  for Anger
    delta_emotion = np.absolute(np.reshape(np.float128(african_matrix[african_emotion_indices[0],9]), (700,1)) - np.reshape(np.float128(european_matrix[european_emotion_indices[0],9]), (700,1)))
    avg_delta_emotion = np.average(delta_emotion)
    print("average delta anger:",avg_delta_emotion)

    # Demographic Parity
    african_emotion_prediction_indices = np.array(np.where(african_matrix[:,8]== emotion_label))
    african_emotion_prediction_count = african_emotion_prediction_indices.shape[1]    
    prob_emotion_african = african_emotion_prediction_count/african_matrix.shape[0]
    print("african "+str(emotion)+ "= ",prob_emotion_african)

    european_emotion_prediction_indices = np.array(np.where(european_matrix[:,8]==emotion_label))
    european_emotion_prediction_count = european_emotion_prediction_indices.shape[1]
    prob_emotion_european = european_emotion_prediction_count/european_matrix.shape[0]
    print("european "+str(emotion)+ "= ",prob_emotion_european)    
    return

#### Read EEC  ####
eec_prediction_file = open('data/eec_predictions/eec_emotion_only_bert.csv')
eec_prediction_csvreader = csv.reader(eec_prediction_file)
eec_prediction_header = next(eec_prediction_csvreader)
eec_prediction_rows = []

for i in eec_prediction_csvreader:
        eec_prediction_rows.append(i)
        
eec_prediction_rows = np.array(eec_prediction_rows)  
#eec_prediction_rows_df = pd.DataFrame(eec_prediction_rows, columns = ['ID','Sentence','Template','Person','Gender','Race'
                                                   #   ,'Emotion', 'Emotion_word', 'Predictions', 'Prediction_Intensity' ])

african_indices = np.where(eec_prediction_rows[:,5] =='African+AC0-American')
african_matrix = eec_prediction_rows[african_indices[0],:] 

european_indices = np.where(eec_prediction_rows[:,5] =='European')
european_matrix = eec_prediction_rows[european_indices[0],:] 

african_predictions_only= african_matrix[:,9]
european_predictions_only= european_matrix[:,9]

african_list = african_predictions_only.tolist()
african_list = [float(i) for i in african_list]

european_list = european_predictions_only.tolist()
european_list = [float(i) for i in european_list]

#Student t-test for RMSE
#https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
value, pvalue = ttest_ind(african_list, european_list, equal_var=False)
print("significance test values:", value, pvalue)


fairness_measures(african_matrix, european_matrix, 'anger', '0')
fairness_measures(african_matrix, european_matrix, 'fear', '1')
fairness_measures(african_matrix, european_matrix, 'joy', '2')
fairness_measures(african_matrix, european_matrix, 'sadness', '3')





