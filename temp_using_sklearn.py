# Linear Regression Model for temperature
from sklearn import model_selection
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random

# y=mx+c this linear equation
# F= 1.8*C+32 this is real time data calculation

# creating data from random values
x = list(range(0,10))                # C
#y = [1.8*F+32  for F in x]      # F use it for fixed values 
y = [1.8*F+32 + random.randint(-4,3) for F in x]
print(f"X : {x}")
print(f"Y : {y}")
plt.plot(x,y,'-*b')

# reshape data for sklearn library
x= np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
# split data
xTrain,xTest,yTrain,yTest= model_selection.train_test_split(x,y,test_size=0.20)

# create linear regression model from linear model of sklearn
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
# print 1.8 and 32 that is coefficent and intercept
print(f'Coefficient: {model.coef_}')
print(f'Intercept : {model.intercept_}')
accuracy=model.score(xTest,yTest)
print(f'Accuracy : {accuracy*100}')

# reconstructed plot
x = x.reshape(1,-1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y = [m*F+c for F in x]
plt.plot(x,y,'-*r')
plt.show()





