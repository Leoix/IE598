# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:08:31 2019

@author: Leo
"""



#list 2-1, 2-2


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


sns.set()


#read data from uci data repository
#List 2-1 count row and cols

data = pd.read_csv('../housing2.csv')
rows = data.shape[0]
columns = data.shape[1]

data2 = data[data['MEDV'].isnull()]
data1 = data.dropna(axis= 0)



columns2 = data2.shape[1]
rows2 = data2.shape[0]

for i in range(columns2-13):
    if data2.sum()[i+13] > np.abs(5 * (data1.mean()[i+13])) or (data.iloc[:,i+13].isnull().any()):
        data2.iloc[:,i+13] = round((data1.mean()[i+13]),1)

data = data1.append(data2)

X = data.iloc[:,:26]
y = data.iloc[:,26]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

reg = LinearRegression()

reg.fit(X_train,y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
coef = reg.coef_
y_intercept = reg.intercept_
R_square = reg.score(X_test,y_test)

plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Trainning data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

print ('The coefficients are ',coef)
print ('The intercept is ', y_intercept)
print ('The R-sqaure is ',R_square)
print ('The MSE of train is ', mean_squared_error(y_train,y_train_pred))
print ('The MSE of test is ', mean_squared_error(y_test,y_test_pred),'\n')

result = []

for i in range(20):
    ridge = Ridge(alpha = 0.05*(i+1),normalize = True)
    ridge.fit(X_train,y_train)
    ridge_train_pred = ridge.predict(X_train)
    ridge_test_pred = ridge.predict(X_test)
    score = ridge.score(X_test,y_test)
    coef = ridge.coef_
    y_intercept = ridge.intercept_
    mse = mean_squared_error(y_test,y_test_pred)
    temp = []
    temp.append(0.1*(i+1))
    temp.append(score)
    temp.append(mse)
    temp.append(coef)
    temp.append(y_intercept)
    result.append(temp)

result.sort(key = lambda x : x[1],reverse = True)
final_result = result[0]

ridge = Ridge(final_result[0],normalize = True)
ridge.fit(X_train,y_train)
ridge_train_pred = ridge.predict(X_train)
ridge_test_pred = ridge.predict(X_test)

print ('The Ridge model coefficients are ',final_result[3])
print ('The Ridge model intercept is ', final_result[4])
print ('The Ridge model R-sqaure is ',final_result[1])
print ('The Ridge model MSE of test is ', mean_squared_error(y_test,ridge_test_pred),'\n')

plt.scatter(ridge_train_pred,ridge_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Trainning data')
plt.scatter(ridge_test_pred,ridge_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Ridge Predicted values')
plt.ylabel('Ridge Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()


result2 = []

for i in range(20):
    lasso = Lasso(alpha = 0.05*(i+1),normalize = True)
    lasso.fit(X_train,y_train)
    lasso_train_pred = lasso.predict(X_train)
    lasso_test_pred = lasso.predict(X_test)
    score = lasso.score(X_test,y_test)
    coef = lasso.coef_
    y_intercept = lasso.intercept_
    mse = mean_squared_error(y_test,y_test_pred)
    temp = []
    temp.append(0.1*(i+1))
    temp.append(score)
    temp.append(mse)
    temp.append(coef)
    temp.append(y_intercept)
    result2.append(temp)

result2.sort(key = lambda x : x[1],reverse = True)
final_result2 = result2[0]

lasso = Lasso(final_result2[0],normalize = True)
lasso.fit(X_train,y_train)
lasso_train_pred = lasso.predict(X_train)
lasso_test_pred = lasso.predict(X_test)

print ('The Lasso model coefficients are ',final_result2[3])
print ('The Lasso model intercept is ', final_result2[4])
print ('The Lasso model R-sqaure is ',final_result2[1])
print ('The Lasso model MSE of test is ', mean_squared_error(y_test,ridge_test_pred))

plt.scatter(lasso_train_pred,lasso_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Trainning data')
plt.scatter(lasso_test_pred,lasso_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Lasso Predicted values')
plt.ylabel('Lasso Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()

# Summary：
# In this exerciese, I will process the data through find all the rows that has a nan in the label feature,
# then I replace the values from the same rows to the mean if it is way off the mean. After that, I performed couple
# visulizations to have a better understanding of the data. Then I conducted Linear regression, Ridge regression, Lasso regression
# on the data with alpha choose from range 0.05 to 1. I find out that for Ridge and Lasso the R2 generally decreases as the alpha increases
# Additionally, even though Lasso drop many unrelated the variables, the R2 is still lower than the ridge regression maybe due to the way I
# replace the outlier data with mean for nan rows 

print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

# # Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW4

