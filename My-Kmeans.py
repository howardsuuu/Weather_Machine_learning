# Unsupervised --- K - means
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

x = [2, 3.5, 6, 9, 6.7, 7.7]
y = [1.1, 6, 3.6, 5.5, 2.8, 7]
# plt.scatter(x, y)
# plt.show()

# arr_x = np.array(x)
# arr_y = np.array(y)

arr_xy = np.c_[x, y]
print(arr_xy)
kmeans = KMeans(n_clusters=2) # 2 == the # of the centroid
kmeans.fit(arr_xy) 

centroids = kmeans.cluster_centers_ 
# will be two centroids because we set 2 in n_cluster=2
label = kmeans.labels_
# labels 0 means belong to the first Centroids, 1 means belong to the second Centroids
# print(centroids)
# print(label)

# ----------Ploting------------------
colors = ["g.", "r.", "c.", "y."]
# colors' position 0 is green, 1 is red
for i in range(len(arr_xy)):
    print("Coordinate: ", arr_xy[i], "Lable: ", label[i])
    plt.plot(arr_xy[i][0], arr_xy[i][1], colors[label[i]], markersize = 5)
    # [i][0] means the first # in the number i array
    # [i][1] means the second # in the number i array

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s = 150, linewidths=5, zorder = 10)
# [:, 0] means all # in the first dimension
# [:, 1] means all # in the second dimension
# s = 150 is the maker's size
plt.show() 

