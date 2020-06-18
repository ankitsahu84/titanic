# Titanic: Machine Learning from Disaster
## Train data characterstics
titanic = pd.read_csv("train.csv")

### display shape of titanic
print(titanic.shape)
(891, 12)

### check if there are duplicate rows, it has 0 rows implying there are no dplicates
dplicateRows = titanic[titanic.duplicated()]
print(dplicateRows.shape)
(0, 12)

No duplicates found

### check data types of each column
print(titanic.dtypes)

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
print(titanic.isnull().sum())

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

## Test data characterstics
### read data from test.csv file into a pandas dataframe
test_titanic = pd.read_csv("test.csv")

### display shape of titanic
print(test_titanic.shape)
(418, 11)

Notice that we have only 11 columns, we need to predict the survived column of this data set
### check data types of each column
print(test_titanic.dtypes)

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

### check if there are duplicate rows, it has 0 rows implying there are no dplicates
dplicateRowsT = test_titanic[test_titanic.duplicated()]
print(dplicateRowsT.shape)
(0, 11)

No duplicates found

#checck for missing values

print(test_titanic.isnull().sum())

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

The first basic analysis is of survived against not survived

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_distribution.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/Variable_distribution.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_by_age.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_distribution_for_pclass.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_by_gender_embarked.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_distribution_survived.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_distribution_vs_embarked.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/age_gender_plcass_embarked.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_geneder_fare.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_per_class.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/embarked_distribution.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_rate_by_class_geneder.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/gender_age_survived.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survival_vs_dependents.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/gender_pclass_survived.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survived_age_boxplot.png) 
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/pclass_distribution.png)
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survived_vs_siblings.png)
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/survval_per_embarked.png)
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/pclass_fare_survived.png)
![alt text](https://github.com/ankitsahu84/titanic/blob/master/Plots/correlation_plot.png) 
