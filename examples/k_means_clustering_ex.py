import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset: Heights and Weights of individuals
data = np.array([[150, 45], [160, 55], [170, 60], [180, 70], [190, 75], [175, 62], [165, 58]])

# Apply K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()