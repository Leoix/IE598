# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 00:06:14 2019

@author: Leo
"""




import numpy as np
import pylab
import scipy.stats as stats
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#read data from uci data repository
#List 2-1 count row and cols

data = pd.read_csv('hw5_treasury yield curve data.csv')
rows = data.shape[0]
columns = data.shape[1]

# data['Adj_Close'].fillna((data['Adj_Close'].mean()), inplace=True)



# print ("Number of Rows of Data which contain NA in target feature= ", rows2)
# print ("Number of Columns of Data which contain NA in target feature= ", columns2)

for i in range(columns-1):
    data.iloc[:,i+1].fillna(data.iloc[:,i+1].mean(),inplace = True)



X = data.iloc[:,1:31]
y = data.iloc[:,31]
y_mod = y.pct_change()
y_mod.fillna(0, inplace=True)
y_mod = y_mod >=0

X_train,X_test,y_train,y_test = train_test_split(X,y_mod,test_size = 0.15,random_state = 42)


model1 = PCA()
model1.fit(X_train)
vr1 = model1.explained_variance_ratio_
print ('The original explained variance ratio is ', vr1)

model2 = PCA(n_components = 3)
X_train_pca = model2.fit_transform(X_train)
vr2 = model2.explained_variance_ratio_
X_test_pca = model2.fit_transform(X_test)
print ('The 3-components explained variance ratio is ', vr2)

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

score1in =lr.score(X_train,y_train)
score1out = lr.score(X_test,y_test)
# RMSE1in = np.sqrt(mean_squared_error(y_train,y_train_pred))
# RMSE1out = np.sqrt(mean_squared_error(y_train,y_pred))

lr.fit(X_train_pca,y_train)
y_pred_pca = lr.predict(X_test_pca)
y_train_pred_pca = lr.predict(X_train_pca)
score2in = lr.score(X_train_pca,y_train)
score2out =lr.score(X_test_pca,y_test)
# RMSE1in = np.sqrt(mean_squared_error(y_test,y_train_pred_pca))
# RMSE1out = np.sqrt(mean_squared_error(y_test,y_pred_pca))

print ('The Logistic in sample R2 score of original data set is ',score1in,' and out of sample R2 is ',score1out)
print ('The Logistic in sample R2 score of PCA transformed data set is ',score2in,' and out of sample R2 is ',score2out)

svm = SVC()
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
y_train_pred = svm.predict(X_train)

score3in =svm.score(X_train,y_train)
score3out = svm.score(X_test,y_test)
# RMSE1in = np.sqrt(mean_squared_error(y_train,y_train_pred))
# RMSE1out = np.sqrt(mean_squared_error(y_train,y_pred))

svm.fit(X_train_pca,y_train)
y_pred_pca = svm.predict(X_test_pca)
y_train_pred_pca = svm.predict(X_train_pca)
score4in = svm.score(X_train_pca,y_train)
score4out =svm.score(X_test_pca,y_test)

print ('The SVM in sample R2 score of original data set is ',score3in,' and out of sample R2 is ',score3out)
print ('The SVM in sample R2 score of PCA transformed data set is ',score4in,' and out of sample R2 is ',score4out)

# Since the adj_close is a continous feature and not able to be fed into the classifier directly. I map the y into True and False
# base on its daily percentage change. The rationale support my transformation is that we want to predict next day's price based on
# today's factor.
# The PCA method reforms the data and left the significant features. It has improved the in sample accuracy for both models.
# And it does improved the out of sample accuracy on the Logistic model. However, the accuracy score somehow decreased under SVC model.
# probably due to overfitting.


print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

print ("Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW5")
