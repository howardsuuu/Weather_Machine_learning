# Supervised ----- Linear Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# -------Not Relevent to this project---------------
# import k-means Clusting from another file called k_meansCl.py
# Clusting() is the class object
import other_Kmeans
k_mean = other_Kmeans.Clustering()
#----------------------------------------------------

regr = linear_model.LinearRegression()
# a = np.arange(8) ---> create an array length of 8
# a = a.reshape(4, 2) ---> shape it to the form of 4 X 2 ( Row X Col  

# -----------------!!!!!!!!Supervised Learning!!!!!!!-----------------
house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
new_size = np.array(size).reshape(-1, 1)
# reshape(row, col)
# -1 means we don't know the dimension of that array, numpy will figure it out itself.
a = regr.fit(new_size, house_price)
print("Coeff: ", regr.coef_)
print("Intercept: ",regr.intercept_)

size_input = 1400
price = (size_input * regr.coef_) + regr.intercept_
# print(price)
# same function different method
# print(regr.predict([[size_input]]))
# -------------Linear Regression-------------------
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    # eval() can transform string to the original type, 
    # Ex: "[1,2,3]" ---> [1, 2, 3] or "1-1" ---> 1-1
graph("regr.coef_ * x + regr.intercept_", range(1000, 2700))
# show the regression (Above)
plt.scatter(size, house_price, color = 'red')
plt.title("")
plt.xlabel("Size of house")
plt.ylabel("Price")
plt.show()
# -----------------------------------------

# ---------------Unsupervised Learning------------
# k-means 
# https://medium.com/@chih.sheng.huang821/機器學習-集群分析-k-means-clustering-e608a7fe1b4