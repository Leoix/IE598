# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:24:40 2019

@author: Leo
"""


import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('C:/Users/LeoCh/OneDrive/Desktop/Treasury Squeeze test - DS1.csv',header=None)

print( 'The scikit learn version is {}.'.format(sklearn.__version__))

X = df.iloc[1:,2:11]
y = df.iloc[1:,11]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)

print( X_train.shape, y_train.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_depth=8)
dt.fit(X_train,y_train)
yt_pred= dt.predict(X_test)
tree_score = accuracy_score(y_test,yt_pred)

k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    yn_pred = knn.predict(X_test)

    scores.append(accuracy_score(y_test,yn_pred))

    

print("The accuracy score of decision tree with a max_depth = 8 is ",tree_score)
print("The accuracy score of KNN with different K is given below")
plt.plot(k_range,scores)

print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
# Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW2