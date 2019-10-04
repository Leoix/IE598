# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:24:40 2019

@author: Leo
"""


import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score,cross_validate

df = pd.read_csv('ccdefault.csv',header=None)


X = df.iloc[1:,1:24]
y = df.iloc[1:,24]

in_sample_score_list = []
out_sample_score_list = []


for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=i)
    dt = DecisionTreeClassifier(max_depth=8)
    dt.fit(X_train,y_train)
    yt_pred= dt.predict(X_test)
    yt_pred_train = dt.predict(X_train)
    in_sample_score = accuracy_score(y_test,yt_pred)
    in_sample_score_list.append(in_sample_score)
    out_sample_score = accuracy_score(y_train,yt_pred_train)
    out_sample_score_list.append(out_sample_score)



print ("The in sample accuracy scores ares: ")
in_sample_score_list = list(np.around(np.array(in_sample_score_list),2))
print (in_sample_score_list)

print ("The out of sample accuracy scores ares: ")
out_sample_score_list = list(np.around(np.array(out_sample_score_list),2))
print (out_sample_score_list)

in_sample_mean = np.round(np.mean(in_sample_score_list),2)
in_sample_std = np.round(np.std(in_sample_score_list),3)
out_sample_mean = np.round(np.mean(out_sample_score_list),2)
out_sample_std = np.round(np.std(out_sample_score_list),3)

print ("In/out   mean     std")
print ("  In    ", in_sample_mean,"  ",in_sample_std)
print ("  Out   ", out_sample_mean,"  ",out_sample_std)


cv = cross_validate(dt,X,y,cv=10,return_train_score=True)
In_sample_score = np.round(cv['train_score'],2)
out_sample_score = np.round(cv['test_score'],2)

print ('CV in sample accuracy score:  %s' %In_sample_score)
print ('CV in sample accuracy score: %.3f +- %.3f' %(np.mean(In_sample_score),np.std(In_sample_score)))
print ('CV out of sample accuracy score:  %s' %out_sample_score)
print ('CV out of sample accuracy score: %.3f +- %.3f' %(np.mean(out_sample_score),np.std(out_sample_score)))
print ("In/out   mean     std")
print ("  In    ", np.round(np.mean(In_sample_score),2),"  ",np.round(np.std(In_sample_score),2))
print ("  Out   ", np.round(np.mean(out_sample_score),2),"  ",np.round(np.std(out_sample_score),2))


# In this exercise, I found out that the train_test_split produces the best out of sample accuaracy, while
# cross_validate is more efficient to run, and the output is summarized in a dictionary. Also the cross_validate
# is just one-line that will do the job, so personally, I think cross_validate is a better practive if the
# difference between accuarcy is not significant.

print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
# Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW6