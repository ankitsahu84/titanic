# Titanic: Machine Learning from Disaster
## Train data characterstics
t_train = pd.read_csv("train.csv")

t_test = pd.read_csv("test.csv")

### display shape of titanic
print(t_train.shape)

(891, 12)

print(t_test.shape)

(418, 11)

### check if there are duplicate rows, it has 0 rows implying there are no dplicates
print(t_train[t_train.duplicated()].shape)

(0, 1@)

print(t_test[t_test.duplicated()].shape)

(0,11)

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

# Some feature engineering now

For some variables I did re-engieer them.

Name field was used to create a title field which was then converted to ordnials for following values

t_train['Title'].value_counts()

Out[334]: 

Mr        517
Miss      182
Mrs       125
Master     40
Rare       27

SibSp and PArch were used to calculate the FamilySize and isAlone features.

Fare was used to create field Fareband so that ordnial values can be used.

#Sklean's KNN Imputer was used to impute the missing values for age, while Embarked was filled with mode in train data and Fare was filled with mode in test data.


# Feture selection

Now that we have a lot of features, we should first see which values are directly related to Survived. Chisquare method was used to get the results.

## Univariate analysis of variables using chi square

chival = pd.DataFrame({'Features': X.columns,'ChiVal':0,'p-Val':0})

chival['ChiVal'] = chi2(X,y)[0]

chival['p-Val'] = chi2(X,y)[1]

print(chival.sort_values(by='ChiVal', ascending=False))

Output was :
     | Features |      ChiVal  |       p-Val|
5         Fare  4518.319091  0.000000e+00
7        Title   228.035979  1.598345e-51
1          Sex   170.348127  6.210585e-39
2          Age    66.239265  3.993769e-16
10    FareBand    39.931983  2.629619e-10
0       Pclass    30.873699  2.753786e-08
9      isAlone    14.640793  1.300685e-04
6     Embarked    11.353117  7.532146e-04
4        Parch    10.097499  1.484707e-03
3        SibSp     2.581865  1.080942e-01
8   Familysize     0.336787  5.616897e-01

Based on the Chival and p-Val I decided to let go of Parch, SibSp and FamilySize features. Also, I deleted Fare feature as I already have Fareband

Not its time for us to check multicolinearity of variables, using VIF. ANything which has VIF more than 5 will be deleted

# Variable multicolinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame({'Features': X.columns,'vif':0,})

vif['vif'] = [variance_inflation_factor(X[X.columns].values, X.columns.get_loc(var)) for var in X.columns]

print(vif.sort_values(by='vif', ascending=False))

Output:
   Features       vif
2       Age  5.804881
6  FareBand  5.361501
0    Pclass  4.213635
5   isAlone  3.326613
4     Title  2.234059
1       Sex  2.142192
3  Embarked  1.359368

Based on the VIF values I deleted Age and Fareband and now the VIF is as below

   Features       vif
0    Pclass  3.344212
4   isAlone  2.502779
1       Sex  2.059545
3     Title  1.986556
2  Embarked  1.345120

# Transform the data using sklearn's SatndardScalar method.
## before we split the data, lets scale the X data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

### tansform the t_test values
t_test = sc.fit_transform(t_test)

# Buld the models

##Post feature selection all the classifiers and scorers were imported

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

#following method was defined so that we can generate a score table for each model

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
    
## All the models were rin and the scores were printed

# Lets run simple logisctic model

lm = LogisticRegression()

modelScores = modelScores.\
    append(pd.Series(performClassification('Logistic Regression-l2',lm,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


# lets run KNN

knn = KNeighborsClassifier(n_neighbors=3)

modelScores = modelScores.\
    append(pd.Series(performClassification('K Nearest Neighbours',knn,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


# lets run GaussianNB


gnb = GaussianNB()

modelScores = modelScores.\
    append(pd.Series(performClassification('Naive Gaussian',gnb,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


# Lets run Decision Tree

dct = DecisionTreeClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('Decision Tree',dct,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)


# Random Forrest


rf = RandomForestClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('Random Forest',rf,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

# Perceptron

pc = Perceptron()

modelScores = modelScores.\
    append(pd.Series(performClassification('Perceptron',pc,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

# stochastic Gradient Decent

sgd = SGDClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('SGD Classifier',sgd,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)

# Artificial neural network

ann = MLPClassifier()

modelScores = modelScores.\
    append(pd.Series(performClassification('ANN',ann,X,y, X_train, y_train, X_test, y_test),\
                     index=modelScores.columns), ignore_index=True)
                     
                     
# print the model scores 

print(modelScores.sort_values(by='CV', ascending=False))

Output:
                     Name      CV Accuracy  Recall Precision      F1 Roc_Auc
7                     ANN   86.44   85.475  69.014    92.453  79.032  82.655
4           Random Forest  85.494   84.916  67.606    92.308  78.049  81.951
0  Logistic Regression-l2  84.812   82.123  73.239      80.0  76.471  80.601
3           Decision Tree  83.505   84.358  66.197    92.157  77.049  81.247
2          Naive Gaussian  82.988   76.536  76.056    68.354    72.0  76.454
6          SGD Classifier  79.256   75.978  49.296    83.333  61.947  71.407
1    K Nearest Neighbours  78.757   82.682  66.197    87.037    75.2  79.858
5              Perceptron  74.999   78.771  67.606     76.19  71.642  76.858

The Ann model was selected and final calculations were done to generate the submission file

finalModel = ann.fit(X,y)

submission['Survived'] = finalModel.predict(t_test).astype(int)

submission.to_csv('submission.csv', index=False)


# With this submission I stand at (top 10%) 2126 out of 23640 participant at an accuracy scoore of .79904. 

![alt text](https://github.com/ankitsahu84/titanic/blob/master/Screen%20Shot%202020-07-01%20at%2010.42.22%20PM.png) 

# More to explore
- Stratified split of data.
- Data imputation based on median of the groups for age (age comes out to be insignificant)
- Can the Cabin be used in some ways to model
