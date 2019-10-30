import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

x = [2, 3.5, 6, 9, 6.7, 7.7]
y = [1.1, 6, 3.6, 5.5, 2.8, 7]
z = [0, 1, 2, 3, 4, 5]

# fArray = np.c_[x, y, z]
# ff = np.array(list(zip(x, y, z)))
# print(ff,'Type is: ',type(ff))
# gg = np.array(zip(x, y, z))
# print(gg,'Type is: ', type(gg))
# print(fArray,'Type is: ',type(fArray))
# print(fArray[:, 2])

test = np.c_[x, y]
print(test)
newTest = np.zeros(test.shape)
print(newTest)