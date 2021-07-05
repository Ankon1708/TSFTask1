# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:06:16 2021

@author: ankon
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('TASK-1 DATA.csv')

#print(df.to_string()) 

print("Info on training examples:")
print(df.info())

x=np.array(df["Hours"]).reshape((-1,1))
#print(x)
y=np.array(df["Scores"])
#print(y)

#Training and Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Training the model
model=LinearRegression().fit(x_train,y_train)

#Testing the model
y_test_model=model.predict(x_test)
y_test_model_error=[]
print("Testing Data:\nHours\tActual Marks\tPredicted Marks\tPercentage Error")
for i in range(len(x_test)):
    y_test_model_error.append(round((y_test[i]-y_test_model[i])*100/y_test[i],3))
    print(x_test[i][0],"\t",y_test[i],"\t\t\t",round(y_test_model[i],3),"\t",y_test_model_error[i])

a=model.intercept_
b=model.coef_[0]
print("\nHypothesis: Y=",round(a,2),"+",round(b,2),"*X")

R2=round(model.score(x_train,y_train),2)
print("Corelation Co-efficient: ",R2)

xpredict=[[9.25]]
ypredict=model.predict(xpredict)
print("Predicted Marks when ",xpredict[0][0]," hours studied: ",round(ypredict[0],0))

#Plotting
plt.scatter(x,y)
plt.plot(x,a+b*x,c="green")
plt.plot(xpredict,ypredict,marker='X',c="black")
plt.show()