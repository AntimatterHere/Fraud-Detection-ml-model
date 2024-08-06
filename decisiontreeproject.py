import pandas as pd
import numpy as np
data = pd.read_csv('creditcard.csv')
print(data.head())
# checking for the null values in dataset
print(data.isnull().sum())
# let’s have a look at the type of transaction
print(data.type.value_counts())
type = data["type"].value_counts()
transactions = type.index
quantity = type.values
# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))
#transforming the categorical features into numerical
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,"CASH_IN": 3, "TRANSFER": 4,"DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())
#let’s train a classification model
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["type"]=le.fit_transform(data["type"])
data["nameOrig"]=le.fit_transform(data["nameOrig"])
data["nameDest"]=le.fit_tarnsform(data["nameDest"])
# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
#Now let’s train the online payments fraud detection model:
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))