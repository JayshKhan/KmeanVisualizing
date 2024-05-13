"""
@author Jaysh Khan
"""

from time import sleep
import matplotlib.pyplot as plt
import random

# making fig global variables
fig, ax = plt.subplots()


def generate_data(num_points, num_features, min_value, max_value):
    data = []
    for _ in range(num_points):
        point = [random.uniform(min_value, max_value) for _ in range(num_features)]
        data.append(point)
    return data


def kmeans(data, num_clusters, max_iters):
    # Initialize centroids randomly
    centroids = random.sample(data, num_clusters)
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

        # animate the clusters with a pause between old and new centroids
        animate_kmeans(clusters, centroids, new_centroids)
        sleep(1)
        centroids = new_centroids


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
        ax.scatter(old_centroids[i][0], old_centroids[i][1], color=colors[i], marker='o', alpha=0.5, s=100)
        for point in cluster:
            ax.scatter(point[0], point[1], color=colors[i])

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


# Generate random data
num_points = 50
num_features = 2
min_value = 0
max_value = 10
data = generate_data(num_points, num_features, min_value, max_value)

num_clusters = 3
max_iters = 100

# Run k-means clustering
kmeans(data, num_clusters, max_iters)

plt.show()
