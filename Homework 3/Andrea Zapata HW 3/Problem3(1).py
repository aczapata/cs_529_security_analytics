import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """

    C = np.zeros((k, X.shape[1]))
    np.random.seed(23)
    ix = np.random.choice(X.shape[0], 1)
    C[0] = X[ix, :]
    D = np.sum(np.square(X - C[0]), axis=1)
    for ki in range(1, k):
        ix = np.argmax(D)
        C[ki] = X[ix, :]
        D = np.minimum(D, np.sum(np.square(X - C[ki]), axis=1))
    return C


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    C = k_init(X, k)

    for i in range(max_iter):
        data_map = assign_data2clusters(X, C)
        for ki in range(k):
            # I multiply the data map column of the given center to get the points in the cluster and remove 0 rows
            cluster_points = (X.T * data_map[:, ki]).T
            cluster_points = cluster_points[~(cluster_points == 0).all(1)]
            C[ki] = np.mean(cluster_points, axis=0)
    return C


def map_cluster(x):
    return [1 if c == np.min(x) else 0 for c in x]


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    k = C.shape[0]
    D = np.zeros((X.shape[0], k))

    for ki in range(k):
        D[:, ki] = np.sum(np.square(X - C[ki]), axis=1)

    data_map = np.apply_along_axis(map_cluster, axis=1, arr=D)
    return data_map


def silhouette_score(xi, ci, k, C_points):
    """ Compute the silhouette score for xi
    ----------
    xi: array, shape(d,)
        Input array d features

    ci: int
        Cluster index of xi

    k: int
        Number of clusters


    C_points: dictionary
        The points in each cluster

    Returns
    -------
    accuracy: float
        The silhouette_score for the given assigments
    """
    if k == 1 :
        return  0
    else:
        bi = sys.maxsize
        ai = sys.maxsize

        for ki in range(k):
            if ki == ci:
                ai = np.mean(np.sum(np.square(C_points[ki] - xi), axis=1))

            else:
                bi = min(bi, np.mean(np.sum(np.square(C_points[ki] - xi), axis=1)))

        si = (bi - ai) / max(bi, ai)
        return si


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assignments
    """
    data_map = assign_data2clusters(X, C)
    C_points = {}

    for ki, ci in enumerate(C):
        cluster_points = (X.T * data_map[:, ki]).T
        cluster_points = cluster_points[~(cluster_points == 0).all(1)]
        C_points[ki] = cluster_points

    silhouette_score_avg = np.zeros(C.shape[0])

    for ki, ci in enumerate(C):
        silhouette_score_avg[ki] = 0
        for xi in C_points[ki]:
            silhouette_score_avg[ki] += silhouette_score(xi, ki, C.shape[0], C_points)
        silhouette_score_avg[ki] /= len(C_points[ki])

    return silhouette_score_avg


# Read dataset
data = pd.read_csv("iris.data", header=None)
data.describe()
# Create new features
x1 = data[0]/data[1]
x2 = data[2]/data[3]
le = preprocessing.LabelEncoder()
le.fit(data[4])
cl = le.transform(data[4])
# Show features with assigned labels
colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(2,2, figsize= (15,15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax[0][0].scatter(x1, x2, c=cl, cmap=matplotlib.colors.ListedColormap(colors))
ax[0][0].set_title("Data distribution")

X = np.column_stack((x1, x2))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
accuracy = np.empty(5)
for k in range(1,5):
    C = k_means_pp(X, k, 5)
    accuracy[k] = np.mean(compute_objective(X, C))

ax[0][1].plot(range(1,5), accuracy[1:])
ax[0][1].set_title("Silhouette score by k")
ax[0][1].set_xlabel("k")
ax[0][1].set_ylabel("silhouette score")

# i will choose k=3 because it has the highest silhouette score
k= 3
accuracy = np.empty(5)
iterations = [1, 2, 5, 10, 50]
for ix, it in enumerate(iterations):
    C = k_means_pp(X, k, it)
    accuracy[ix] = np.mean(compute_objective(X, C))

ax[1][0].plot(iterations, accuracy)
ax[1][0].set_title("silhouette score by # of iterations")
ax[1][0].set_xlabel("# of iterations")
ax[1][0].set_ylabel("silhouette score")

C = k_means_pp(X, k, 5)
data_map = assign_data2clusters(X, C)
C_points = {}
for ki in range(k):
    cluster_points  = (X.T * data_map[:,ki]).T
    cluster_points = cluster_points[~(cluster_points==0).all(1)]
    C_points[ki] = cluster_points

ax[1][1].scatter(C_points[0][:,0], C_points[0][:,1])
ax[1][1].scatter(C_points[1][:,0], C_points[1][:,1], c='green')
ax[1][1].scatter(C_points[2][:,0], C_points[2][:,1], c='orange')
ax[1][1].scatter(C[:,0], C[:,1], c='red')
ax[1][1].set_title("Cluster centers and data distribution k=3")
plt.show()
