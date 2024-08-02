# K-means Clustering Visualization Project

This project implements a K-means clustering algorithm with data visualization using Matplotlib and SciPy. The project includes the following key components:

## Components

1. **Data Generation**: 
   - The `generate_data` function creates random data points with specified features and value ranges.

2. **Elbow Method**: 
   - The `elbow_method` function determines the optimal number of clusters by plotting the cost/inertia for different values of \( k \).

3. **K-means Clustering**: 
   - The `kmeans` function performs the K-means clustering algorithm, iteratively updating centroids and assigning data points to clusters until convergence or a maximum number of iterations is reached.

4. **Visualization**: 
   - The `animate_kmeans` function visualizes the clustering process by plotting the data points, centroids, and cluster boundaries at each iteration, saving the plots as images before updating.

5. **Execution**: 
   - The script generates random data, runs the Elbow method to determine the optimal number of clusters, performs K-means clustering, and displays the clustering process using Matplotlib.

## Files

- `kmeans_visualizing_without_boundary.py`: Implements the K-means clustering algorithm with data visualization.
- `kmeans_visualizing_with_boundary.py`: Implements the K-means clustering algorithm with data visualization and cluster boundary.
## Usage

1. **Generate Data**: 
   - Use the `generate_data` function to create random data points.

2. **Determine Optimal Clusters**: 
   - Use the `elbow_method` function to determine the optimal number of clusters.

3. **Run K-means Clustering**: 
   - Use the `kmeans` function to perform the clustering.

4. **Visualize Clustering Process**: 
   - Use the `animate_kmeans` function to visualize and save the clustering process.

## Requirements

- Python 3.x
- Matplotlib
- SciPy

## Installation

Install the required packages using pip:

```sh
pip install matplotlib scipy
```

## Running the Project

1. Generate random data.
2. Determine the optimal number of clusters using the Elbow method.
3. Perform K-means clustering.
4. Visualize the clustering process.

```sh
python3 kmean_visualizing_without_boundary.py

python3 kmean_visualilzing_with_boundary.py

```

## License

This project is licensed under the MIT License.