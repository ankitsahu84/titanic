#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 12:05:47 2020

@author: Ankit Sahu

Objective: To predict the Survival of passengers on Titanic ship based on passenger attributes.

"""

#Import the required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#checck for missing values
print(test_titanic.isnull().sum())

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















