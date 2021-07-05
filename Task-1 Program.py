# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:06:16 2021

@author: ankon
"""

#%%Importing modules and the data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('TASK-1 DATA.csv')

#print(df.to_string()) 


print("Info on dataset:")
print(df.info())
print("First 5 samples:\n",df.head())
print("Last 5 samples:\n",df.tail())

x=np.array(df["Hours"]).reshape((-1,1))
#print(x)
y=np.array(df["Scores"])
#print(y)

#%%Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#Training the model
model=LinearRegression().fit(x_train,y_train)

#Testing the model
y_test_model=model.predict(x_test)
y_test_model_error=[]
print("Testing Data:\nHours\tActual Marks\tPredicted Marks\t\tPercentage Error")
for i in range(len(x_test)):
    y_test_model_error.append(round((y_test[i]-y_test_model[i])*100/y_test[i],3))
    print(x_test[i][0],"\t",y_test[i],"\t\t\t",round(y_test_model[i],3),"\t\t\t",y_test_model_error[i])

#%%Presenting the results
a=model.intercept_
b=model.coef_[0]
print("\nHypothesis: Y=",round(a,2),"+",round(b,2),"*X")

x_predict=[[9.25]]
y_predict=model.predict(x_predict)
print("Predicted Marks when ",x_predict[0][0]," hours studied: ",round(y_predict[0],0))

R2=round(model.score(x_train,y_train),3)
print("Corelation Co-efficient: ",R2)

#%%Visualization
plt.scatter(x,y)
plt.plot(x,a+b*x,c="green")
plt.plot(x_predict,y_predict,marker='X',c="black")
plt.show()