
"""
We should load the dataset from iris_onehot.csv and create clusters.
This is an alternative to breaking the features into quartiles in order to then binarize them.

This lets us say something like:
When we cluster the data into a cluster of 2, we now have a binary feature for each cluster. Called 1of2, 2of2, etc.
With a cluster of 3, we have 3 binary features. Called 1of3, 2of3, 3of3, etc.
This is a good way to cluster the data and then binarize the features.

When a new data point comes in, we can find its binary value for 1of2, 2of2, 1of3, 2of3, 3of3, etc.

Eventually, we will want to identify poorly performing features and remove them from the dataset.
Additionally, we identify poorly performing subsets and try to create clusters for them.
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the one-hot encoded dataset
iris_onehot_df = pd.read_csv('datasets/iris_onehot.csv')
print(iris_onehot_df.head())

# Load the one-hot dataset
iris_df = pd.read_csv('datasets/iris_onehot.csv')
print(iris_df.head())

species_df = pd.read_csv('datasets/iris.csv')
print(species_df.head())

X = iris_df.drop(columns=['target_0', 'target_1', 'target_2'])
y = iris_df[['target_0', 'target_1', 'target_2']]

feature_names = X.columns
target_names = ['target_0', 'target_1', 'target_2'] # Use original target labels
print("Feature names:", feature_names)
print("Target names:", target_names)
print(X.shape, y.shape)
print(X.head())
print(y.head())

# 2. Scale Data (Important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)





# 3. Apply KMeans with k=2
k = 2
kmeans2 = KMeans(n_clusters=k, # Set the desired number of clusters
                n_init='auto', # Suppresses a future warning, uses smart initialization
                random_state=42) # For reproducible results
kmeans2.fit(X_scaled)
# 4. Get Results
cluster_labels2 = kmeans2.labels_ # Array assigning each sample to a cluster (0 or 1)
cluster_centers2 = kmeans2.cluster_centers_ # Coordinates of the cluster centroids (in scaled space)
for i, center in enumerate(cluster_centers2):
    print(f"Cluster {i} center: {center}")
print("Cluster labels:", cluster_labels2)
print("Cluster labels shape:", cluster_labels2.shape)

# k=3
k = 3
kmeans3 = KMeans(n_clusters=k, # Set the desired number of clusters
                n_init='auto', # Suppresses a future warning, uses smart initialization
                random_state=42) # For reproducible results
kmeans3.fit(X_scaled)
# 4. Get Results
cluster_labels3 = kmeans3.labels_ # Array assigning each sample to a cluster (0 or 1)
cluster_centers3 = kmeans3.cluster_centers_ # Coordinates of the cluster centroids (in scaled space)
for i, center in enumerate(cluster_centers3):
    print(f"Cluster {i} center: {center}")
print("Cluster labels:", cluster_labels3)
print("Cluster labels shape:", cluster_labels3.shape)


# Find the centroids in the original space (unscaled)
k2_centroids_unscaled = scaler.inverse_transform(cluster_centers2)
print("Unscaled cluster centers (k=2):")
print(k2_centroids_unscaled)
k3_centroids_unscaled = scaler.inverse_transform(cluster_centers3)
print("Unscaled cluster centers (k=3):")
print(k3_centroids_unscaled)
"""
Unscaled cluster centers (k=2):
[[6.262 2.872 4.906 1.676]
 [5.006 3.428 1.462 0.246]]
Unscaled cluster centers (k=3):
[[6.78085106 3.09574468 5.5106383  1.97234043]
 [5.006      3.428      1.462      0.246     ]
 [5.80188679 2.67358491 4.36981132 1.41320755]]
"""

# Visualize
# fig of k2 clusters
df = pd.DataFrame(X, columns=feature_names)
df['Original Class'] = [target_names[i] for i in y.values.argmax(axis=1)]
df['KMeans (k=2) Cluster'] = cluster_labels2
print("\nDataFrame with Original Classes and KMeans (k=2) Clusters (first 5 rows):")
print(df.head())
# Optional: Compare clustering results to original classes
# Since we force k=2 on 3 original classes, there will be overlap.
# This shows how many points from each original class ended up in each new cluster.
print("\nCross-tabulation of Original Classes vs KMeans (k=2) Clusters:")
print(pd.crosstab(df['Original Class'], df['KMeans (k=2) Cluster']))
# Optional: Visualize (using first two features for simplicity)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels2, cmap='viridis', alpha=0.7, edgecolors='k')
# Plot centroids
centers = plt.scatter(cluster_centers2[:, 0], cluster_centers2[:, 1], c='red', s=200, alpha=0.9, marker='X')
plt.title('Iris Dataset Clustered into k=2 using KMeans (Scaled Features)')
plt.xlabel(f"Scaled {feature_names[0]}")
plt.ylabel(f"Scaled {feature_names[1]}")
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1'])
plt.savefig('datasets/iris_kmeans_2.png')


# fig of k3 clusters
df = pd.DataFrame(X, columns=feature_names)
df['Original Class'] = [target_names[i] for i in y.values.argmax(axis=1)]
df['KMeans (k=3) Cluster'] = cluster_labels3
print("\nDataFrame with Original Classes and KMeans (k=3) Clusters (first 5 rows):")
print(df.head())
# Optional: Compare clustering results to original classes
# Since we force k=3 on 3 original classes, there will be overlap.
# This shows how many points from each original class ended up in each new cluster.
print("\nCross-tabulation of Original Classes vs KMeans (k=3) Clusters:")
print(pd.crosstab(df['Original Class'], df['KMeans (k=3) Cluster']))
# Optional: Visualize (using first two features for simplicity)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels3, cmap='viridis', alpha=0.7, edgecolors='k')
# Plot centroids
centers = plt.scatter(cluster_centers3[:, 0], cluster_centers3[:, 1], c='red', s=200, alpha=0.9, marker='X')
plt.title('Iris Dataset Clustered into k=3 using KMeans (Scaled Features)')
plt.xlabel(f"Scaled {feature_names[0]}")
plt.ylabel(f"Scaled {feature_names[1]}")
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1'])
plt.savefig('datasets/iris_kmeans_3.png')

# plt.show()

# one-hot encode the cluster labels
cluster_labels2_onehot = pd.get_dummies(cluster_labels2, prefix='K2').astype(int)
print("\nOne-hot encoded cluster labels (k=2):")
print(cluster_labels2_onehot.head())
cluster_labels3_onehot = pd.get_dummies(cluster_labels3, prefix='K3').astype(int)
print("\nOne-hot encoded cluster labels (k=3):")
print(cluster_labels3_onehot.head())
# save the cluster labels to a new dataframe
df = pd.DataFrame(X, columns=feature_names)

df['KMeans (k=2) Cluster'] = cluster_labels2
df['KMeans (k=3) Cluster'] = cluster_labels3
df = pd.concat([df, cluster_labels2_onehot, cluster_labels3_onehot], axis=1)
df['target_0'] = y['target_0']
df['target_1'] = y['target_1']
df['target_2'] = y['target_2']

df.to_csv('datasets/iris_kmeans-2-3_clusters.csv', index=False)
print("\nDataFrame with Original Classes and KMeans Clusters (first 5 rows):")
print(df.head())

"""
DataFrame with Original Classes and KMeans Clusters (first 5 rows):
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  KMeans (k=2) Cluster  KMeans (k=3) Cluster  K2_0  K2_1  K3_0  K3_1  K3_2  target_0  target_1  target_2
0                5.1               3.5                1.4               0.2                     1                     1     0     1     0     1     0         1         0         0      
1                4.9               3.0                1.4               0.2                     1                     1     0     1     0     1     0         1         0         0      
2                4.7               3.2                1.3               0.2                     1                     1     0     1     0     1     0         1         0         0      
3                4.6               3.1                1.5               0.2                     1                     1     0     1     0     1     0         1         0         0      
4                5.0               3.6                1.4               0.2                     1                     1     0     1     0     1     0         1         0         0      
"""
