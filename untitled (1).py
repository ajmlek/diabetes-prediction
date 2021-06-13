import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


df= pd.read_csv("C:\\Users\hp\Desktop\diabetes.csv")

df.isnull().sum()

df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

from sklearn.model_selection import train_test_split
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20)
classifier.fit(x_train,y_train)

filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
