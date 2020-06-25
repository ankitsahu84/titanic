#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 12:05:47 2020

@author: Ankit Sahu

Objective: To predict the Survival of passengers on Titanic ship based on passenger attributes.
https://www.kaggle.com/startupsci/titanic-data-science-solutions

"""

#Import the required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('max_columns', 12)

#change the working directory 
os.chdir("/Users/ankit/CU/kaggle/titanic")

#read data from test.csv file into a pandas dataframe
titanic = pd.read_csv("train.csv")

#display shape of titanic
print(titanic.shape)

#display 5 records
print(titanic.head(5))

#check data types of each column
print(titanic.dtypes)

#check if there are duplicate rows, it has 0 rows implying there are no dplicates
dplicateRows = titanic[titanic.duplicated()]
print(dplicateRows.shape)

del dplicateRows

#checck for missing values
print(titanic.isnull().sum())

#now check the test data for similar things
#read data from test.csv file into a pandas dataframe
test_titanic = pd.read_csv("test.csv")

#display shape of titanic
print(test_titanic.shape)

#display 5 records
print(test_titanic.head(5))

#check data types of each column
print(test_titanic.dtypes)

#check if there are duplicate rows, it has 0 rows implying there are no dplicates
dplicateRowsT = test_titanic[test_titanic.duplicated()]
print(dplicateRowsT.shape)

del dplicateRowsT
#checck for missing values
print(test_titanic.isnull().sum())


#plot graphs
fig = plt.figure(figsize=(18,6))  

#normalized % of survival rate
plt.figure(0)
plt.title("Survival distribution")
titanic.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#normalized plot of survival 
plt.figure(1)
plt.title("Passenger Class distribution")
titanic.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#scatter plot of survival wrt Age
plt.figure(2)
plt.title("Age distribution for survived values")
plt.scatter(titanic.Survived, titanic.Age, alpha=0.1)


#line chart of age wrt to class
plt.figure(4)
for x in [1,2,3]:    ## for 3 classes
    titanic.Age[titanic.Pclass == x].plot(kind="kde")
plt.title("Age wrt Pclass")
plt.legend(("1st","2nd","3rd"))

#normalized plot of Embarked 
plt.figure(5)
plt.title("Embarked  distribution")
titanic.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#line chart of age wrt to embarked
plt.figure(3)
for x in ['S','C','Q']:    ## for 3 classes
    titanic.Age[titanic.Embarked == x].plot(kind="kde")
plt.title("Age wrt Embarked")
plt.legend(("S","C","Q"))

plt.figure(6)
#histogram of all variables
titanic.hist(bins=10,figsize=(9,7),grid=False);

#hist of sex, age and survived 
plt.figure(7)
g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="green");

#hist of survived, sex and Pclass
plt.figure(8)
g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Pclass",color="purple");

#hist of pclass and survived fare and age
plt.figure(9)
g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True, palette={1:"seagreen", 0:"red"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

#hist of fare, age, sex and survived
plt.figure(10)
g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender and Fare');


plt.figure(12)
#factor plor of survived per location
sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="g").fig.suptitle("How many survived per embarked location");

plt.figure(13)
#factorplot for survived based on Pclass
sns.factorplot(x = 'Pclass',y="Survived", data = titanic,color="gray").fig.suptitle("How many survived per Pclass");

plt.figure(14)
#factor plot for survived per Sex
sns.factorplot(x = 'Sex',y="Survived", data = titanic,color="black").fig.suptitle("How many survived per Gender");

plt.figure(15)
#factorplot for survived per sibsp
sns.factorplot(x = 'SibSp',y="Survived", data = titanic,color="orange").fig.suptitle("Survival vs siblings");

plt.figure(16)
#factor plot for survived per Parch
sns.factorplot(x = 'Parch',y="Survived", data = titanic,color="b").fig.suptitle("Survival vs dependents");


#How many Men and Women Survived by Passenger Class
plt.figure(17)
sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=titanic, saturation=.5,
                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class');


#How many Men and Women Survived by embark location
plt.figure(11)
sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Embarked",
                    data=titanic, saturation=.5,
                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Embarked Location');



#Survival distribution by age
plt.figure(18)
ax = sns.boxplot(x="Survived", y="Age", 
                data=titanic);

plt.figure(19)
ax = sns.stripplot(x="Survived", y="Age",
                   data=titanic, jitter=True,
                   edgecolor="gray")
plt.title("Survival by Age",fontsize=12);


plt.figure(22)
g = sns.factorplot(x="Age", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=titanic[titanic.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5, 
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2);
plt.subplots_adjust(top=0.8)
g.fig.suptitle("Age, Gender, Embarked and class distribution");

#corrrelation plot
plt.figure(21)
corr=titanic.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');

#correlation of features with target variable
titanic.corr()["Survived"]


#method to split the x and y values for data for training, cv and test
def getTrainCVtestData(inputData, y):
    #return 60% data as training, 20% as test and 20% as cross validation
    
    #shuffle the data set
    newDf = inputData.sample(frac=1).reset_index(drop=True)
    
    datarow = inputData.shape[0]
    trainSize = round(.6 * datarow)
    cvSize = round(.2 * datarow)
    
    xTrain = newDf.loc[0:trainSize,newDf.columns != y]
    yTrain = newDf.loc[0:trainSize,newDf.columns == y]
    
    xCV = newDf.loc[trainSize+1:trainSize+cvSize,newDf.columns != y]
    yCV = newDf.loc[trainSize+1:trainSize+cvSize,newDf.columns == y]
   
    xTest = newDf.loc[trainSize+cvSize+1:datarow,newDf.columns != y]
    yTest = newDf.loc[trainSize+cvSize+1:datarow,newDf.columns == y]
    
    return xTrain, yTrain, xCV, yCV, xTest, yTest


#transform the Cabin feature to Deck, where deck is the first character of the cabin
titanic['Cabin'] = titanic['Cabin'].str[:1]

#first fill in Na values with U
titanic['Cabin'] = titanic['Cabin'].fillna('U')

#fill in embarked values with 'S' as it is the most common value
titanic['Embarked'] = titanic['Embarked'].fillna('S')

#change male/female to binary values
titanic['Sex'] = titanic['Sex'].replace({'male': 0, 'female': 1})

#Transform fare
titanic['Fare'] = np.log(titanic['Fare'] + 1)

#create a copy of data for basic model
bTitanic = titanic

#remove PassengerId column, it is not needed
del bTitanic['PassengerId']

#remove ticket column, it is not needed
del bTitanic['Ticket']

#remove Name column, it is not needed
del bTitanic['Name']

#process the categorical valyes
titanic_cat = titanic[['Embarked','Cabin']].iloc[:]

#convert categorical values of Embarked and Cabin to dummines
bTitanicE = pd.get_dummies(titanic_cat.astype('str'))

#remove the categorical columns
del bTitanic['Cabin']
del bTitanic['Embarked']

#concatenate the new columns to the bTitanic 
bTitanic = pd.concat([bTitanic,bTitanicE], axis=1)

#remove objects that are not needed.
del bTitanicE
del titanic_cat

print(bTitanic.isnull().sum())


#Lets start with simplest model to predict. We will use the above method to split data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate   

from sklearn.impute import KNNImputer

xTrain, yTrain, xCV, yCV, xTest, yTest = getTrainCVtestData(bTitanic, 'Survived')

#impute Age value
imputer = KNNImputer(n_neighbors=3)

xTrain = pd.DataFrame(data = imputer.fit_transform(xTrain),
                                 columns = xTrain.columns,
                                 index = xTrain.index) 

xCV = pd.DataFrame(data = imputer.fit_transform(xCV),
                                 columns = xCV.columns,
                                 index = xCV.index) 

xTest = pd.DataFrame(data = imputer.fit_transform(xTest),
                                 columns = xTest.columns,
                                 index = xTest.index) 

titanic1 = titanic
del titanic1['Survived']
del titanic1['Name']
titanic1 = pd.DataFrame(data = imputer.fit_transform(titanic1),
                                 columns = titanic1.columns,
                                 index = titanic1.index) 

#prepare data for cross validations
y = bTitanic['Survived']

del bTitanic['Survived']

X = bTitanic

#impute missing values
X = pd.DataFrame(data = imputer.fit_transform(X),
                                 columns = X.columns,
                                 index = X.index) 




# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xTrain = sc.fit_transform(xTrain)
xCV = sc.fit_transform(xCV)
X_xTesttest = sc.transform(xTest)

X = sc.transform(X)

#Lets try a very basic model with logistic regression
lm = LogisticRegression()
lm_model = lm.fit(xTrain, yTrain.values.ravel())

#lets calculate the predicted values of lm for xCV set
y_lm_pred_cv = lm_model.predict(xCV)

y_lm_prob_cv = lm_model.predict_proba(xCV)

lm_cv_score = accuracy_score(yCV, y_lm_pred_cv)

#lets calculate the predicted values of lm for xCV set
y_lm_pred_test = lm_model.predict(xTest)

y_lm_prob_test = lm_model.predict_proba(xTest)

lm_test_score = accuracy_score(yTest, y_lm_pred_test)

#cross validation and its scopre
lm_cross_validation = cross_val_score(lm, X,y.values.ravel(), cv=10)



















#lets check the how the training performed on lm
lm_accuracy = accuracy_score(yCV,y_lm_pred)
print(lm_accuracy)

#lets check cross validation score
lm_cv_score = cross_val_score(lm, xCV,yCV.values.ravel(), cv=10)
print(cv_score.mean())

#recall
lm_recall_score = recall_score(yCV, y_lm_pred)
print(lm_recall_score)

#precision
lm_precision_score = precision_score(yCV, y_lm_pred)
print(lm_precision_score)

#f1 scrore
lm_f1_score= f1_score(yCV, y_lm_pred)
print(lm_f1_score)

#ROC score
lm_roc_score = roc_auc_score(yCV, y_lm_prob[:,1])
print(lm_roc_score)










