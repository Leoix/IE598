

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#show the shape of dataset
data=pd.read_csv("ccdefault.csv")
X = data.iloc[:,1:24]
y = data.iloc[:,24]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)


n_estimators_list = [50,150,300,500]

for i in n_estimators_list:
    rfc = RandomForestClassifier(criterion= 'gini',n_estimators = i)
    rfc.fit(X_train,y_train)
    y_train_pred = rfc.predict(X_train)
    y_pred = rfc.predict(X_test)
    in_sample_accuracy = accuracy_score(y_train,y_train_pred)
    out_sample_accuracy = accuracy_score(y_test,y_pred)
    print ("The in sample accuracy score of ",i," estimator of Random Forest Model is ", in_sample_accuracy)
    print ("The out sample accuracy score of ",i," estimator of Random Forest Model is ", out_sample_accuracy)

rfc = RandomForestClassifier(criterion = 'gini', n_estimators = 300)
rfc.fit(X_train,y_train)
importance_rfc = pd.Series(rfc.feature_importances_,index = X.columns)
sorted_importance_rf = importance_rfc.sort_values()

sorted_importance_rf.plot(kind = 'barh',color = 'lightgreen');
plt.show()

print ("In general,thre is no clear linear relationship between n_estimator and in sample accuracy, however, the")
print ("computation time is greatly increased when n_estimator increased.The optimal n_estimator of mine model is 300")
print ("The top 3 most important features in my model is Pay_0, Age, and Bill_AMT1")
print ("The concept of feature important is how much the tree nodes use a particular feature (weighted average) to reduce impurity")
print ("It is calculated as the decrease in node impurity weighted by the probability of reaching that node")
print ("\n")
print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW7")