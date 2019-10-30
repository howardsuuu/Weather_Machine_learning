from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 9)

data = pd.read_csv(r"/Users/howardsu666/Github/pyythonn/ML_project/"
                   r"weather_data.csv", engine='python')
# print(data.shape)
# print out dimension / here is (3000, 2)
dx = data['Temp Diff'].values
# values return in np.array
dy = data['Pressure Diff'].values
xy = np.array(list(np.c_[dx, dy]))
# or np.c_[dx, dy]
# or np.array(list(zip(dx, dy)))
# plt.scatter(dx, dy, c = 'black', s=8)
# 
# ----------歐幾里德距離--------
def distance(a, b, ax = 1):
    return np.linalg.norm(a - b, axis=ax)
# ----------# of cluster------
k =3 
centroids_x = np.random.randint(0, np.max(xy)-20, size=3)
# X coordinates of random centroids
centroids_y = np.random.randint(0, np.max(xy)-20, size=3)

centroids = np.array(list(zip(centroids_x, centroids_y)), dtype=np.float32)
# print(centroids)

# --------PLotting centroids---and data itself----
plt.scatter(dx, dy, c='black', s = 8)
plt.scatter(centroids_x, centroids_y, marker='*', s=200, c='r')
# plt.show()

# --------store centroids value when it Updates---------
centroid_old = np.zeros(centroids.shape)
# makes all elements in array zero, based on its original dimesion
# (Same shape as Centroids' array)

# Cluster labels(0, 1, 2)
clusters = np.zeros(len(xy))
# makes it [0, 0, 0, .....0] base on it length, only one dimension

# distance between old and new centroids
error = distance(centroids, centroid_old, None)
while error != 0:
    for i in range(len(xy)):
        distances = distance(xy[i], centroids)
        # Distance from xy's [i] coordinate to the centroid

        cluster = np.argmin(distances)
        
# https://www.geeksforgeeks.org/numpy-argmin-python/
# return the smallest number's indices (position)
        clusters[i] = cluster

# ---------Store old centroids value------------
    centroid_old = deepcopy(centroids)

# ---------Finding new centroids by taking avg value-----
     
# Number of cluster
    for i in range(k):
        points = [xy[j] for j in range(len(xy)) if clusters[j] == i]
        centroids[i] = np.mean(points, axis=0)
    error = distance(centroids, centroid_old, None)

colors = ['g', 'b', 'r', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([xy[j] for j in range(len(xy)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], s=200, marker = '*', c='black')
plt.show()