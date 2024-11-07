import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("creditcard.csv")
# data=data.drop(["Unnamed: 5","Unnamed: 6"])
print(data.head())

print(data.isnull().sum())

# Exploring transaction type
print(data.type.value_counts())

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
            values=quantity, 
            names=transactions,hole = 0.5, 
            title="Distribution of Transaction Type")
figure.show()

newbalanceOrig=list(data["newbalanceOrig"])
errors=["C865699625"]
for i in range(len(newbalanceOrig)):
  if newbalanceOrig[i]==errors[0]:
    newbalanceOrig[i]=0
data["newbalanceOrig"]=newbalanceOrig

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
#data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())

# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))

# splitting the data
from sklearn.model_selection import train_test_split
#x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
#y = np.array(data[["isFraud"]])
x=data.iloc[:,:4].values
y=data.iloc[:,-1].values
print(x)
#print(y)

from math import nan
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=nan, strategy='mean')
y=y.reshape(-1,1)
imputer.fit(y)
y = imputer.transform(y).reshape(-1)
print(y)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=42)
classifier.fit(X_train,y_train)

#predicting test sets:

y_pred=lr.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


