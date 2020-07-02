#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:02:50 2020

@author: Ankit Sahu

"""
#Import the required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#change the working directory 
os.chdir("/Users/ankit/CU/kaggle/titanic")

#read data from test.csv file into a pandas dataframe
t_train = pd.read_csv("train.csv")
t_test = pd.read_csv("test.csv")

#create a submission array which will hold the passenger ids of the values we need to hold
submission = pd.DataFrame({"PassengerId": t_test["PassengerId"],'Survived': 0 })

#Lets delete the the non relevant columns
del t_train['PassengerId'], t_train['Ticket']
del t_test['PassengerId'], t_test['Ticket']

#LETS Perform some EDA
#display shape of titanic
print(t_train.shape)
print(t_test.shape)

#display 5 records
print(t_train.head(5))
print(t_test.head(5))

#check data types of each column
print(t_train.dtypes)
print(t_test.dtypes)

#check if there are duplicate rows, it has 0 rows implying there are no dplicates
print(t_train[t_train.duplicated()].shape)
print(t_test[t_test.duplicated()].shape)

#checck for missing values
print(t_train.isnull().sum())
#checck for missing values
print(t_test.isnull().sum())


#plot graphs
fig = plt.figure(figsize=(18,6))  

#normalized % of survival rate
plt.figure(0)
plt.title("Survival distribution")
t_train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#normalized plot of survival 
plt.figure(1)
plt.title("Passenger Class distribution")
t_train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#scatter plot of survival wrt Age
plt.figure(2)
plt.title("Age distribution for survived values")
plt.scatter(t_train.Survived, t_train.Age, alpha=0.1)


#line chart of age wrt to class
plt.figure(4)
for x in [1,2,3]:    ## for 3 classes
    t_train.Age[t_train.Pclass == x].plot(kind="kde")
plt.title("Age wrt Pclass")
plt.legend(("1st","2nd","3rd"))

#normalized plot of Embarked 
plt.figure(5)
plt.title("Embarked  distribution")
t_train.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

#line chart of age wrt to embarked
plt.figure(3)
for x in ['S','C','Q']:    ## for 3 classes
    t_train.Age[t_train.Embarked == x].plot(kind="kde")
plt.title("Age wrt Embarked")
plt.legend(("S","C","Q"))

plt.figure(6)
#histogram of all variables
t_train.hist(bins=10,figsize=(9,7),grid=False);

#hist of sex, age and survived 
plt.figure(7)
g = sns.FacetGrid(t_train, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="green");

#hist of survived, sex and Pclass
plt.figure(8)
g = sns.FacetGrid(t_train, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Pclass",color="purple");

#hist of pclass and survived fare and age
plt.figure(9)
g = sns.FacetGrid(t_train, hue="Survived", col="Pclass", margin_titles=True, palette={1:"seagreen", 0:"red"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();

#hist of fare, age, sex and survived
plt.figure(10)
g = sns.FacetGrid(t_train, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender and Fare');


plt.figure(12)
#factor plor of survived per location
sns.factorplot(x = 'Embarked',y="Survived", data = t_train,color="g").fig.suptitle("How many survived per embarked location");

plt.figure(13)
#factorplot for survived based on Pclass
sns.factorplot(x = 'Pclass',y="Survived", data = t_train,color="gray").fig.suptitle("How many survived per Pclass");

plt.figure(14)
#factor plot for survived per Sex
sns.factorplot(x = 'Sex',y="Survived", data = t_train,color="black").fig.suptitle("How many survived per Gender");

plt.figure(15)
#factorplot for survived per sibsp
sns.factorplot(x = 'SibSp',y="Survived", data = t_train,color="orange").fig.suptitle("Survival vs siblings");

plt.figure(16)
#factor plot for survived per Parch
sns.factorplot(x = 'Parch',y="Survived", data = t_train,color="b").fig.suptitle("Survival vs dependents");


#How many Men and Women Survived by Passenger Class
plt.figure(17)
sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=t_train, saturation=.5,
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
                    data=t_train, saturation=.5,
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
                data=t_train);

plt.figure(19)
ax = sns.stripplot(x="Survived", y="Age",
                   data=t_train, jitter=True,
                   edgecolor="gray")
plt.title("Survival by Age",fontsize=12);


plt.figure(22)
g = sns.factorplot(x="Age", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=t_train[t_train.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5, 
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2);
plt.subplots_adjust(top=0.8)
g.fig.suptitle("Age, Gender, Embarked and class distribution");

#corrrelation plot
plt.figure(21)
corr=t_train.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');

#correlation of features with target variable
t_train.corr()["Survived"]

#EDA Ends

import re
#define a function to calculate the title using reges
def calcTitle(name):
    return re.search(", (.*?)\.",name).group(1)
    
#call the function for all rows 
t_train['Title'] = t_train['Name'].apply(calcTitle)

t_test['Title'] = t_test['Name'].apply(calcTitle)

#delete unsed variables
del t_train['Name'], t_test['Name']
del t_train['Cabin'], t_test['Cabin']

t_train['Embarked'] = t_train['Embarked'].fillna(t_train['Embarked'].mode()[0])    
t_test['Fare'] = t_test['Fare'].fillna(t_test['Fare'].mode()[0]) 

#transofmr the categorical values of Title
from feature_engine import categorical_encoders as ce

rare_encoder = ce.RareLabelCategoricalEncoder(tol=0.04489,n_categories=3)

t_train['Title'] = rare_encoder.fit_transform(pd.DataFrame(t_train['Title']))
t_test['Title'] = rare_encoder.fit_transform(pd.DataFrame(t_test['Title']))

del rare_encoder

t_train['Familysize'] = t_train['SibSp'] + t_train['Parch'] + 1
t_test['Familysize'] = t_test['SibSp'] + t_test['Parch'] + 1

t_train['isAlone'] = np.where((t_train['Familysize'] > 1),0,t_train['Familysize'])
t_test['isAlone'] = np.where((t_test['Familysize'] > 1),0,t_test['Familysize'])

t_train['Sex'] = t_train['Sex'].replace({'male': 0,'female':1}) 
t_test['Sex'] = t_test['Sex'].replace({'male': 0,'female':1})

t_train['Embarked'] = t_train['Embarked'].replace({'S': 0,'C':1,'Q':2})    
t_test['Embarked'] = t_test['Embarked'].replace({'S': 0,'C':1,'Q':2}) 

t_train['Title'] = t_train['Title'].replace({'Mr': 0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})    
t_test['Title'] = t_test['Title'].replace({'Mr': 0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})    

from sklearn.impute import KNNImputer

#impute Fare in test set and for age in train set
imputer = KNNImputer(n_neighbors=3)

t_train = pd.DataFrame(data = imputer.fit_transform(t_train),
                                 columns = t_train.columns,
                                 index = t_train.index) 

t_test = pd.DataFrame(data = imputer.fit_transform(t_test),
                                 columns = t_test.columns,
                                 index = t_test.index) 
del imputer

#create sub grops of age and fareså

t_train['FareBand'] = pd.qcut(t_train['Fare'], 4, labels = [1, 2, 3, 4]).astype(int)
t_test['FareBand'] = pd.qcut(t_test['Fare'], 4, labels = [1, 2, 3, 4]).astype(int)

#feature selection routine
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

y = t_train['Survived']

del t_train['Survived']
X = t_train

#Univariate analysis of variables using chi square
chival = pd.DataFrame({'Features': X.columns,'ChiVal':0,'p-Val':0})
chival['ChiVal'] = chi2(X,y)[0]
chival['p-Val'] = chi2(X,y)[1]

print(chival.sort_values(by='ChiVal', ascending=False))

del t_train['Parch'], t_train['SibSp'],t_train['Familysize'] ,t_train['Fare'] 
del t_test['Parch'], t_test['SibSp'],t_test['Familysize'] ,t_test['Fare']

#Variable multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({'Features': X.columns,'vif':0,})
vif['vif'] = [variance_inflation_factor(X[X.columns].values, X.columns.get_loc(var)) for var in X.columns]
print(vif.sort_values(by='vif', ascending=False))

#vif is high Age and fareband
del t_train['Age'] ,t_test['Age']
del t_train['FareBand'] ,t_test['FareBand']

#before we split the data, lets scale the X data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

#tansform the t_test values
t_test = sc.fit_transform(t_test)

del sc

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import  accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#create a data frame to store the scores
modelScores = pd.DataFrame(columns =['Name','CV','Accuracy','Recall','Precision','F1','Roc_Auc'])

def performClassification(name, estimator, X, y, X_train, y_train, X_test, y_test):
    
    model = estimator.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cv_score = round((cross_val_score(estimator, X ,y.values.ravel(), cv=5, scoring='roc_auc').mean())*100,3)
    
    accuracy = round((accuracy_score(y_test, y_pred))*100,3)
    
    recall = round((recall_score(y_test, y_pred))*100,3)
    
    precision = round((precision_score(y_test, y_pred))*100,3)
    f1 = round((f1_score(y_test, y_pred))*100,3)
    
    roc_auc = round((roc_auc_score(y_test, y_pred))*100,3)

    returnArray = pd.array([name,cv_score,accuracy,recall,precision,f1,roc_auc])
    
    
    return returnArray



#Lets run simple logisctic model
lm = LogisticRegression()
modelScores = modelScores.\
    append(pd.Series(performClassification('Logistic Regression-l2',lm,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#lets run KNN
knn = KNeighborsClassifier(n_neighbors=3)
modelScores = modelScores.\
    append(pd.Series(performClassification('K Nearest Neighbours',knn,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#lets run GaussianNB


gnb = GaussianNB()
modelScores = modelScores.\
    append(pd.Series(performClassification('Naive Gaussian',gnb,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#Lets run Decision Tree

dct = DecisionTreeClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('Decision Tree',dct,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#Random Forrest


rf = RandomForestClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('Random Forest',rf,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#Perceptron
pc = Perceptron()
modelScores = modelScores.\
    append(pd.Series(performClassification('Perceptron',pc,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#stochastic Gradient Decent
sgd = SGDClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('SGD Classifier',sgd,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#Artificial neural network
ann = MLPClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('ANN',ann,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)
        
#RVM or Relevance Vector Machine

#print the model scores 
print(modelScores.sort_values(by='CV', ascending=False))

#0.79904
finalModel = ann.fit(X,y)

submission['Survived'] = finalModel.predict(t_test).astype(int)

submission.to_csv('submission5.csv', index=False)

#Variable multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({'Features': X.columns,'vif':0,})
vif['vif'] = [variance_inflation_factor(X[X.columns].values, X.columns.get_loc(var)) for var in X.columns]
print(vif.sort_values(by='vif', ascending=False))

del t_train['Age']
del t_test['Age']

#before we split the data, lets scale the X data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

#tansform the t_test values
t_test = sc.fit_transform(t_test)

del sc

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import  accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#create a data frame to store the scores
modelScores = pd.DataFrame(columns =['Name','CV','Accuracy','Recall','Precision','F1','Roc_Auc'])

def performClassification(name, estimator, X, y, X_train, y_train, X_test, y_test):
    
    model = estimator.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cv_score = round((cross_val_score(estimator, X ,y.values.ravel(), cv=5, scoring='roc_auc').mean())*100,3)
    
    accuracy = round((accuracy_score(y_test, y_pred))*100,3)
    
    recall = round((recall_score(y_test, y_pred))*100,3)
    
    precision = round((precision_score(y_test, y_pred))*100,3)
    f1 = round((f1_score(y_test, y_pred))*100,3)
    
    roc_auc = round((roc_auc_score(y_test, y_pred))*100,3)

    returnArray = pd.array([name,cv_score,accuracy,recall,precision,f1,roc_auc])
    
    
    return returnArray



#Lets run simple logisctic model
lm = LogisticRegression()
modelScores = modelScores.\
    append(pd.Series(performClassification('Logistic Regression-l2',lm,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#lets run KNN
knn = KNeighborsClassifier(n_neighbors=3)
modelScores = modelScores.\
    append(pd.Series(performClassification('K Nearest Neighbours',knn,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#lets run GaussianNB


gnb = GaussianNB()
modelScores = modelScores.\
    append(pd.Series(performClassification('Naive Gaussian',gnb,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#Lets run Decision Tree

dct = DecisionTreeClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('Decision Tree',dct,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


#Random Forrest


rf = RandomForestClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('Random Forest',rf,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#Perceptron
pc = Perceptron()
modelScores = modelScores.\
    append(pd.Series(performClassification('Perceptron',pc,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#stochastic Gradient Decent
sgd = SGDClassifier()
modelScores = modelScores.\
    append(pd.Series(performClassification('SGD Classifier',sgd,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

#Artificial neural network
ann = MLPClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('ANN',ann,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)
        
#RVM or Relevance Vector Machine

#print the model scores 
print(modelScores.sort_values(by='CV', ascending=False))

finalModel = ann.fit(X,y)

submission['Survived'] = finalModel.predict(t_test).astype(int)

submission.to_csv('submission.csv', index=False)
