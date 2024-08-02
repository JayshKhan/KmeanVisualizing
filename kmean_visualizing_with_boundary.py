"""
The kmean_visualizing_with_boundary.py file implements a K-means clustering algorithm with data visualization
using Matplotlib and SciPy. It includes the following key components:
- Data Generation: The generate_data function creates random data points with specified features and value ranges.
- Elbow Method: The elbow_method function determines the optimal number of clusters by
    plotting the cost/inertia for different values of ( k ).
- K-means Clustering: The kmeans function performs the K-means clustering algorithm, iteratively updating centroids
    and assigning data points to clusters until convergence or a maximum number of iterations is reached.
- Visualization: The animate_kmeans function visualizes the clustering process by plotting the data points, centroids,
    and cluster boundaries at each iteration.
- Execution: The script generates random data, runs the Elbow method to determine the optimal number of clusters,
    performs K-means clustering, and displays the clustering process using Matplotlib.

@author Jaysh Khan
"""
import numpy as np
from scipy.spatial import ConvexHull
from time import sleep
import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans

# making fig global variables so that it should not be declared again and again
fig, ax = plt.subplots()

"""
the function elbow_method determines the optimal number of
 clusters by plotting the cost/inertia for different values of ( k ).
:param data: The data points to cluster.
"""


def elbow_method(data):
    # Elbow Method
    ks = range(1, 6)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(data)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    # Plot ks vs inertias
    fig1, ax1 = plt.subplots()
    ax1.plot(ks, inertias, '-o')
    ax1.set_xlabel('number of clusters, k')
    ax1.set_ylabel('Cost/inertia')
    ax1.set_title('Elbow Method')
    # plt.show()


"""
The generate_data function creates random data points with specified features and value ranges.
:param num_points: The number of data points to generate.
:param num_features: The number of features for each data point.
:param min_value: The minimum value for the data points.
:param max_value: The maximum value for the data points.
:return: A list of random data points.
"""


def generate_data(num_points, num_features, min_value, max_value):
    data = []
    for _ in range(num_points):
        point = [random.uniform(min_value, max_value) for _ in range(num_features)]
        data.append(point)
    return data


"""
The kmeans function performs the K-means clustering algorithm, iteratively updating centroids and
assigning data points to clusters until convergence or a maximum number of iterations is reached.
:param data: The data points to cluster.
:param num_clusters: The number of clusters to create.
:param max_iters: The maximum number of iterations to run the algorithm.
"""


def kmeans(data, num_clusters, max_iters):
    # Initialize centroids randomly
    centroids = random.sample(data, num_clusters)
    costs = []
    print(centroids)
    for _ in range(max_iters):
        clusters = [[] for _ in range(num_clusters)]
        for point in data:
            # calculate the distance of the point from each centroid
            distances = [sum((point[i] - centroid[i]) ** 2 for i in range(len(point))) for centroid in centroids]
            # assign the point to the cluster with the closest centroid
            cluster = distances.index(min(distances))
            clusters[cluster].append(point)

        # calculate new centroids
        new_centroids = [[sum(point[i] for point in cluster) / len(cluster) for i in range(len(point))] for cluster in
                         clusters]
        # check if the centroids have converged
        if all(all(abs(a - b) < 1e-5 for a, b in zip(old, new)) for old, new in zip(centroids, new_centroids)):
            print("Converged")
            break

        costs.append(
            sum([sum((point[i] - new_centroids[cluster][i]) ** 2 for i in range(len(point))) for cluster, points in
                 enumerate(clusters) for point in points]))

        # animate the clusters with a pause between old and new centroids
        animate_kmeans(clusters, centroids, new_centroids)
        sleep(1)
        centroids = new_centroids

    # plot costs array
    plt.figure()
    plt.plot(costs)
    plt.title("Costs")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


"""
The animate_kmeans function visualizes the clustering process by plotting the data points and centroids at
each iteration,
:param clusters: The clusters of data points.
:param old_centroids: The old centroids of the clusters.
:param new_centroids: The new centroids of the clusters.
"""


def animate_kmeans(clusters, old_centroids, new_centroids):
    print(f'Clusters: {clusters}')
    # Show the clusters and centroids
    # plot it on the global axis
    ax.clear()
    fig.suptitle("K-means Clustering")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # clear the axis
    # show the centroids

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(clusters):
        # plot the old centroid of the cluster
        ax.scatter(old_centroids[i][0], old_centroids[i][1], color=colors[i], marker='x', alpha=0.5, s=100)
        for point in cluster:
            ax.scatter(point[0], point[1], color=colors[i])

        # Draw boundary around the cluster
        cluster_points = np.array(cluster)
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color=colors[i])

    plt.pause(1)  # Pause for 1 second

    ax.clear()  # Clear the plot for the new centroids
    fig.suptitle("K-means Clustering")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    for i, cluster in enumerate(clusters):
        # plot the new centroid of the cluster
        ax.scatter(new_centroids[i][0], new_centroids[i][1], color=colors[i], marker='x', s=100)
        for point in cluster:
            ax.scatter(point[0], point[1], color=colors[i])

        # Draw boundary around the cluster
        cluster_points = np.array(cluster)
        hull = ConvexHull(cluster_points)
        for simplex in hull.simplices:
            ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], color=colors[i])


if __name__ == "__main__":
    # Generate random data
    num_points = 50
    num_features = 2
    min_value = 0
    max_value = 10
    data = generate_data(num_points, num_features, min_value, max_value)

    num_clusters = 3
    max_iters = 100

    elbow_method(data)  # Determine the optimal number of clusters then manually set the number of clusters

    # Run k-means clustering
    kmeans(data, num_clusters, max_iters)

    plt.show()
