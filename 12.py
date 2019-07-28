import numpy as np
from scipy.linalg import eigh
import matplotlib
from scipy import linalg

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def data_generation(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30. * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x

class LaplacianEigenmap:
    def __init__(self, n_components, k):
        """
        param: n_component : embedding dim
        param: k : knn of similarity matrix
        """
        self.n_components = n_components
        self.k = k

    def transform(self, X):
        W = self.create_similarity_matrix(X)
        D = self.get_degree_matrix(W)
        D = D.astype(float)
        L = D - W
        eig_val, eig_vec = eigh(L, D)
        eig_vec = eig_vec.T
        index = np.argsort(eig_val)
        eig_vec = eig_vec[index]
        phi = eig_vec[1:self.n_components + 1]
        return phi.T

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_similarity_matrix(self, X):
        """create Similarity matrix (knn)

        :param X: data matrix (data_nX,feature_n)
        """
        W = []
        for x_i in X:
            W.append(self.k_nearest_list(X, x_i))
        W = np.array(W)
        return np.where(np.logical_or(W, W.T), 1, 0)

    def k_nearest_list(self, X, x_i):
        """
        return the ndarray containing bool value represents whether the value is in k nearest neighbor of x_i
        e.g. ndarray [True False True]
        """
        dist_list = [self.dist(x_i, x_j) for x_j in X]
        sorted_list = sorted(dist_list)  # 昇順
        threshold = sorted_list[self.k - 1]
        dist_list = np.array(dist_list)
        knn_list = dist_list <= threshold
        assert sum(knn_list) == self.k, knn_list
        return knn_list

    def dist(self, x_i, x_j):
        return np.dot(x_i - x_j, x_i - x_j)

def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 1], z[:, 0], c=a, marker='o')
    
n = 1000
a, x = data_generation(n)
Laplacian=LaplacianEigenmap(n_components=100, k=8)
z = Laplacian.transform(x)
visualize(x, z, a)
