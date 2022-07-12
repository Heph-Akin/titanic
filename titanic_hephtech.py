# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 08:19:02 2022

@author: HephTech
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot


# Importing the dataset
dataset = pd.read_csv('train.csv')



#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset.Sex = enc.fit_transform(dataset.Sex)



# Creating a variable for number of people travelling together and people travelling alone 
dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['Alone'] = 0
dataset.loc[dataset['Family'] == 1, 'Alone'] = 1

# Creating Dummy Variables for Embarkation Locations
embarked = dataset['Embarked'].str.get_dummies()
dataset = pd.concat([dataset,embarked], axis="columns")




# Selecting dependent and independent variables
#X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16]].values
X = dataset.iloc[:, [2, 4, 5, 9, 12]].values

y = dataset.iloc[:, 1].values




# Spliiting dataset into training and tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25)

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
X_train= imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the classifier to thew dataset
# libraries for decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

# libraries for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Create the classifier
tree = RandomForestClassifier(n_estimators=500)
classifier = tree.fit(X_train, y_train)



# get importance
importance = classifier.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# Predicting the Test set Results 
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




#Testing the Algorithm
# importing Test set
test = pd.read_csv('test.csv')


#Encoding categorical variables
enc = LabelEncoder()
test.Sex = enc.fit_transform(test.Sex)



# Creating a variable for number of people travelling together and people travelling alone 
test['Family'] = test['SibSp'] + test['Parch'] + 1
test['Alone'] = 0
test.loc[test['Family'] == 1, 'Alone'] = 1

# Creating Dummy Variables for Embarkation Locations
embarked_test = test['Embarked'].str.get_dummies()
test = pd.concat([test,embarked_test], axis="columns")

# Selecting dependent and independent variables
#test_set = test.iloc[:, [1,3,4,5,6,8, 11, 12, 13, 14, 15]].values
test_set = test.iloc[:, [1, 3, 4, 8, 11]].values


# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
test_set= imputer.fit_transform(test_set)



#Feature Scaling
test_set = sc_X.fit_transform(test_set)


# Predicting Survival on the testset
test_pred = classifier.predict(test_set)

z = pd.DataFrame(test['PassengerId'])
z = z.astype(np.int32)
test_pred = pd.DataFrame(test_pred)


# Joining PassangerId To Predictions 
test_results = pd.concat([z, test_pred], axis="columns")


test_results.columns = ["PassengerId", "Survived"]

test_results.to_csv('titanic_hephtech.csv', index=False)