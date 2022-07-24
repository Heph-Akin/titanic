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

# Adding a variable for Whether or not passenger has a cabin
dataset['HasCabin'] = ~dataset['Cabin'].isna()

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset.Sex = enc.fit_transform(dataset.Sex)
dataset.HasCabin = enc.fit_transform(dataset.HasCabin)

# =============================================================================
# String Analysis to extract Titles from the Name variable
# =============================================================================

#Adding a variable for titles of people 
dataset['Titles'] = dataset['Name'].str.extract(r'(Mrs\.|Mr\.|Miss\.|Col\.|Dr\.|Major\.|Master\.|Rev\.)')
dataset['Titles'] = dataset['Titles'].fillna("UT")
dataset['Titles'].value_counts()

#Encoding the title variables
# Title Category Codes - Major. - 0, Dr. - 1, Col. - 2, Master - 3, Miss - 4, Mr - 5, Mrs - 6, Rev. - 7, UT(Unknown Title) - 8
dataset['Titles'] = dataset['Titles'].astype('category')
dataset['Title'] = dataset['Titles'].cat.codes


# Creating a variable for number of people travelling together and people travelling alone 
dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['Alone'] = 0
dataset.loc[dataset['Family'] == 1, 'Alone'] = 1


# Creating Dummy Variables for Embarkation Locations
embarked = dataset['Embarked'].str.get_dummies()
dataset = pd.concat([dataset,embarked], axis="columns")



# Selecting dependent and independent variables
X = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'Title', 'Family', 'Alone', 'C', 'Q', 'S']]
y = dataset.iloc[:, 1].values




# Spliiting dataset into training and tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25)


# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
X_train= pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X.columns)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(sc_X.transform(X_test), columns=X.columns)

# Fitting the classifier to the dataset
# libraries for SVM & random forest
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Create the classifiers
#Fitting the random forrest classifier
forest = RandomForestClassifier(n_estimators=1000)
classifier_forest = forest.fit(X_train, y_train)

#Fitting SVM classifier
svm = SVC(kernel='poly') 
classifier_svm = svm.fit(X_train, y_train)


# =============================================================================
# Checking Feature importance and selecting most valuable features impoves model accuracy
# =============================================================================

# getting feature importance
importance = pd.Series(classifier_forest.feature_importances_, index = X.columns)
importance.nlargest(20).plot(kind = 'barh')
resolution_value = 1200
plt.savefig("Feature Imporatance.png", format="png", dpi=resolution_value)


#Feature Selection (Selecting Feature with greater than 0.02 importance)
X = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'Title', 'Family']]
y = dataset.iloc[:, 1].values


#Creating new algorithm based on selected features.
# Spliiting dataset into training and tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25)


# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
X_train= pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X.columns)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = pd.DataFrame(sc_X.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(sc_X.transform(X_test), columns=X.columns)

# Fitting the classifier to the dataset
# libraries for SVM & random forest
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Create the classifiers
#Fitting the random forrest classifier
forest = RandomForestClassifier(n_estimators=1000)
classifier_forest = forest.fit(X_train, y_train)





#Scoring the algorithm
classifier_svm.score(X_test, y_test)

classifier_forest.score(X_test, y_test)

# =============================================================================
# SVM Model has an accuracy of 77%
# Optimized Randopm Forest had a score of 83% 
# =============================================================================


# Predicting the Test set Results 
y_pred = classifier_forest.predict(X_test)



# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



# =============================================================================
# Testing the Algorithm
# =============================================================================

# importing Test set
test = pd.read_csv('test.csv')

# Adding a variable for Whether or not passenger has a cabin
test['HasCabin'] = ~test['Cabin'].isna()

#Encoding categorical variables
enc = LabelEncoder()
test.Sex = enc.fit_transform(test.Sex)
test.HasCabin = enc.fit_transform(test.HasCabin)

# =============================================================================
# String Analysis to extract Titles from the Name variable
# =============================================================================
#Adding a variable for titles of people 
test['Titles'] = test['Name'].str.extract(r'(Mrs\.|Mr\.|Miss\.|Col\.|Dr\.|Major\.|Master\.|Rev\.)')
test['Titles'] = test['Titles'].fillna("UT")
test['Titles'] = test['Titles'].astype('category')
test['TitleCodes'] = test['Titles'].cat.codes


# Creating a variable for number of people travelling together and people travelling alone 
test['Family'] = test['SibSp'] + test['Parch'] + 1
test['Alone'] = 0
test.loc[test['Family'] == 1, 'Alone'] = 1

# Creating Dummy Variables for Embarkation Locations
embarked_test = test['Embarked'].str.get_dummies()
test = pd.concat([test,embarked_test], axis="columns")

# Selecting independent variables
test_set = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'TitleCodes', 'Family']]


# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
test_set= pd.DataFrame(imputer.fit_transform(test_set), columns = test_set.columns)


#Feature Scaling
test_set = pd.DataFrame(sc_X.fit_transform(test_set), columns = test_set.columns)


# Predicting Survival on the testset
test_pred = classifier_forest.predict(test_set)

z = pd.DataFrame(test['PassengerId'])
z = z.astype(np.int32)
test_pred = pd.DataFrame(test_pred)


# Joining PassangerId To Predictions 
test_results = pd.concat([z, test_pred], axis="columns")


test_results.columns = ["PassengerId", "Survived"]

test_results.to_csv('titanic_hephtech.csv', index=False)