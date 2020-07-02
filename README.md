# Titanic: Machine Learning from Disaster
## Train data characterstics
t_train = pd.read_csv("train.csv")

t_test = pd.read_csv("test.csv")

### display shape of titanic
print(t_train.shape)

(891, 10)

print(t_test.shape)

(418, 9)

### check if there are duplicate rows, it has 0 rows implying there are no dplicates
print(t_train[t_train.duplicated()].shape)

(0, 10)

print(t_test[t_test.duplicated()].shape)

(0,9)

No duplicates found

### check data types of each column

print(t_train.dtypes)

| feature | type |
|-----------|----------|
PassengerId | int64
Survived |        int64
Pclass    |       int64
Name       |     object
Sex        |    object
Age         |   float64
SibSp        |    int64
Parch         |   int64
Ticket         | object
Fare      |     float64
Cabin    |       object
Embarked |       object

### #check for missing values
print(t_train.isnull().sum())

| feature | missing values |
|-----------|----------|
PassengerId |     0
Survived    |     0
Pclass      |     0
Name        |     0
Sex         |     0
Age         |   177
SibSp       |     0
Parch       |     0
Ticket      |     0
Fare        |     0
Cabin       |   687
Embarked    |     2

Based on above, we can see there are lot of missing values in Age and Cabin. We need to address these in any model which we use.

Notice that we have only 11 columns, we need to predict the survived column of this data set
### check data types of each column
print(t_test.dtypes)

| feature | type |
|-----------|----------|
PassengerId | int64
Pclass    |       int64
Name       |     object
Sex        |    object
Age         |   float64
SibSp        |    int64
Parch         |   int64
Ticket         | object
Fare      |     float64
Cabin    |       object
Embarked |       object

No anamaly dfound in data types of columns in train and test data

#checck for missing values

print(t_test.isnull().sum())

| feature | missing values |
|-----------|----------|
PassengerId |     0
Pclass      |     0
Name        |     0
Sex         |     0
Age         |    86
SibSp       |     0
Parch       |     0
Ticket      |     0
Fare        |     1
Cabin       |   327
Embarked    |     0

## Lets analyse the relations of variables to the dependent variable 

The basic distribution plots suggest that 40% people survived, Most people embarked from S, and most people were travelling in 3rd class. 

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_distribution.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/pclass_distribution.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/embarked_distribution.png) 

The distribution of variables

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/Variable_distribution.png) 

The age distribution suggest that the majority users who survived were in the age range 10 - 40.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_by_age.png) 

The age distribution agains passenger class and embarked locations tells that for 1st class and location Q the age distribution was almost normal.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_distribution_for_pclass.png)  
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_distribution_vs_embarked.png) 

The survival rate for men and women  agains passenger class and embarked location suggest that women from class 1 and 2 and women from C and Q embarked location survived the most.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_by_gender_embarked.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_rate_by_class_geneder.png) 

The scatter plot of survival and not survival agains passenger class and fare doesnt give much info other than the fact the class 1 passengers paid more fare and aged people survived less in class 1.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_geneder_fare.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/pclass_fare_survived.png)


The gender distribution on age and survival suggests that males between 20 to 50 died more as compared to other segements.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/gender_age_survived.png) ![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/gender_pclass_survived.png) 


The survival distribution for embarked location, passenger class, dependents, siblings. We can see that 55% passengers boarding from C survived. It can also be observed that when sibsp was between 0 to 2 then survival increased. Similaryl, if parch is between 0 to 3, survival rate is higher. 

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_per_class.png) ![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_vs_dependents.png)  ![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survived_vs_siblings.png) ![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survval_per_embarked.png)

It can be seen that age is normally distributed for those who survived.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survived_age_boxplot.png) 

The following heatmap can be reffered to see the distribution of age, embarked location, gender and passenger. The most normally distributed is the passengers in class 1.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_gender_plcass_embarked.png) 

The correlation plot of all variables.

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/correlation_plot.png) 

The individual variable's correaltion with survivale rate.

t_train.corr()["Survived"]

Out[2]: 
| variable | correation |
|-------|---------|
PassengerId |  -0.005007
Survived    |  1.000000
Pclass      |  -0.338481
Age         |  -0.077221
SibSp       | -0.035322
Parch       |  0.081629
Fare        |  0.257307
