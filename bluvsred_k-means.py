import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200);

plt.show()
