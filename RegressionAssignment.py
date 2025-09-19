import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
PART 1: Basic Linear Regression
"""

data = pd.read_csv("RegressionData.csv", header=None, names=['X', 'y'])  

X = data['X'].values.reshape(-1, 1) 
y = data['y']  

plt.scatter(X, y)  

reg = linear_model.LinearRegression()  
reg.fit(X, y) 

fig = plt.figure()
y_pred = reg.predict(X)  
plt.scatter(X, y, c='b')  
plt.plot(X, y_pred, 'r')  
fig.canvas.draw()

print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_[0])  

print("The profit/loss in a city with 18 habitants is ", reg.predict([[18]]))  


"""
PART 2: Logistic Regression
"""

data = pd.read_csv("LogisticRegressionData.csv", header=None, names=['Score1', 'Score2', 'y'])  

X = data[['Score1', 'Score2']]  
y = data['y']  

m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i],
                marker=m[data['y'][i]], color=c[data['y'][i]])  
fig.canvas.draw()

regS = linear_model.LogisticRegression()  
regS.fit(X, y)  

y_pred = regS.predict(X)  
m = ['o', 'x']
c = ['red', 'blue']  
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i],
                marker=m[y_pred[i]], color=c[y_pred[i]]) 
fig.canvas.draw()


"""
PART 3: Multi-class Classification using Logistic Regression
"""

"""
The One-vs-Rest (OvR) method works by training one binary classifier per class. 
Each classifier separates one class (positive) from all other classes (negative). 
For prediction, the classifier that outputs the highest confidence score determines the class label.
"""

"""
The One-vs-One (OvO) method trains one binary classifier for every possible pair of classes. 
For a dataset with N classes, this results in N*(N-1)/2 classifiers. 
During prediction, each classifier votes for a class, and the class with the most votes is selected as the final prediction.
"""
