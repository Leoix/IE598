
#list 2-1, 2-2


import numpy as np
import pylab
import scipy.stats as stats
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot 
import random

#read data from uci data repository
#List 2-1 count row and cols

data = pd.read_csv('HW2Data.csv')
rows = data.shape[0]
columns = data.shape[1]

#arrange data into list for labels and list of lists for attributes
# xList = []
# labels = []
# for line in data:
#     #split on comma
#     row = line.strip().split(",")
#     xList.append(row)


print ("Number of Rows of Data = ", rows)
print ("Number of Columns of Data = ", columns)



#List 2-2 determine numeric or catergorical
darray = data.values
# nrow = len(xList)
# ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(columns):
    for row in range(rows):
        try:
            a = float(darray[row][col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(darray[row][col]) > 0:
                type[1] += 1
            else:
                type[2] += 1

print ("There are ", type[0], " numerical entries and ", type[1], " Strings", "and  ", type[2], " Other entries\n")


    
# #List 2-3 generate summary statistics for column 3 (e.g.)
col = 9
x = darray [0][9]
# for row in xList:
#     colData.append(float(row[col]))
# colArray = np.array(colData) 

couponSum = np.sum(darray,axis = 0)[9]
couponMean = couponSum /(rows)
couponstd = data.std(axis = 0, skipna = True, numeric_only = True)[0]

print ("Mean of coupon is ", couponMean, " and the std of coupon is ", couponstd,"\n")

percentiles1 = [0.25,0.50,0.75,1]
percentiles2 = [0.1,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1]

for i in range(len(percentiles1)):
    coupon_percentiles = data.quantile(percentiles1[i],axis=0,numeric_only = True)[0]
    print ("The ", percentiles1[i]*100, " percentiles of coupon is ", coupon_percentiles)

for i in range(len(percentiles2)):
    coupon_percentiles = data.quantile(percentiles2[i],axis=0,numeric_only = True)[0]
    print ("The ", percentiles2[i]*100, " percentiles of coupon is ", coupon_percentiles)

print ("\n")

#The last column contains categorical variables
col = 30

counts = data['bond_type'].value_counts().to_dict()

keys = list(counts.keys())

for i in range(len(counts.keys())):
    print ("Bond_type ", i, " has ", counts[i+1], " bonds")


coupon_array = list(np.array(np.matrix(darray).T)[9])

#List 2-4 Quantile-Quantile Plot
stats.probplot(coupon_array, dist="norm", plot=pylab)
pylab.show()


# #List 2-5 read csv

#read  data into pandas data frame

#print head and tail of data frame
print(data.head())
print(data.tail())
#print summary of data frame
summary = data.describe()
print(summary) 


# #list 2-6 Parallel Coordinates Graph

pd.plotting.parallel_coordinates(
    data[['Coupon', 'Issued Amount', 'LiquidityScore', 'Maturity Type']],
    'Maturity Type')

plot.show()

# #List 2-7 cross plot pairing
#calculate correlations between real-valued attributes
dataRow2 = data.iloc[:,35]
dataRow3 = data.iloc[:,36]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("36nd Attribute")
plot.ylabel(("37rd Attribute"))
plot.show()


dataRow21 = data.iloc[:,34]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("36nd Attribute")
plot.ylabel(("35st Attribute"))
plot.show()


# #Listing 2-8: Correlation between Classifi cation Target and Real Attributes
target = []
for i in range(2721):
    #assign 0 or 1 target value based on "M" or "R" labels
    if data.iat[i,12] == "DEFAULTED":
        target.append(1.0)
    else:
        target.append(0.0)

#plot 35th attribute
dataRow = data.iloc[:,34]
plot.scatter(dataRow, target)
plot.xlabel("Attribute Value 1")
plot.ylabel("Target Value 1")
plot.show()
# #
# #To improve the visualization, this version dithers the points a little
# # and makes them somewhat transparent
target = []
for i in range(2721):
#assign 0 or 1 target value based on "M" or "R" labels
# and add some dither
    if data.iat[i,12] == "DEFAULTED":
        target.append(1.0 + random.uniform(-0.1, 0.1))
    else:
        target.append(0.0 + random.uniform(-0.1, 0.1))

# #plot 35th attribute with semi-opaque points
dataRow = data.iloc[:,34]
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.xlabel("Attribute Value2")
plot.ylabel("Target Value2")
plot.show()

#Listing 2-9: Pearson’s Correlation Calculation
#calculate correlations between real-valued attributes
dataRow2 = data.iloc[:,20]
dataRow3 = data.iloc[:,21]
dataRow21 = data.iloc[:,35]
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i]/numElt
    mean3 += dataRow3[i]/numElt
    mean21 += dataRow21[i]/numElt

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/numElt

corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * (dataRow3[i] - mean3) / (np.sqrt(var2*var3) * numElt)
    corr221 += (dataRow2[i] - mean2) *  (dataRow21[i] - mean21) / (np.sqrt(var2*var21) * numElt)

print ("Correlation between attribute 2 and 3 is", corr23)
print ("Correlation between attribute 2 and 21 is", corr221)
print (" \n")


#Listing 2-10: Presenting Attribute Correlations Visually
#calculate correlations between real-valued attributes
corMat = DataFrame(data.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()


print("My name is Chenyi Yang")
print("My NetID is: cyang75")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

# Chenyi Yang Github URL: https://github.com/Leoix/IE598_F18_HW1/HW3

