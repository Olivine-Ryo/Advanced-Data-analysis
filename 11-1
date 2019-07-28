import numpy as np
import matplotlib
from scipy.linalg import eig

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(5)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y,n_components=1):
    w, v = eig(Sample_between_matrix(x,y),Sample_within_matrix(x,y))
    
    index=np.argsort(w)
    w = w[index]
    v = v[index]
    """Fisher Discriminant Analysis.
    Implement this function

    Returns
    -------
    T : (1, 2) ndarray
        The embedding matrix.
    """
    return v[n_components:, :]

def Sample_between_matrix(x,y):
    S=0
    for y_i in set(y):
        mu = np.mean(x[y==y_i],axis=0, keepdims=True)
        S += np.dot(mu.T,mu) * np.count_nonzero(y == y_i)
    return S

def Sample_within_matrix(x,y):
    S=0
    for y_i in set(y):
        norm_x = x- x[y==y_i].mean(axis=0)
        S += np.dot(norm_x.T,norm_x)
        
        #doing same thing
        
        #for i in range(norm_x.shape[0]):
        #    S += np.dot(np.array([norm_x[i]]).T,np.array([norm_x[i]]))
        
        print(S)
    return S

def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 9,
             np.array([-T[:, 1], T[:, 1]]) * 9, 'k-')
    plt.legend()
    plt.savefig('lecture11-h1.png')

n_components = 1
sample_size = 100
#x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y, n_components)
visualize(x, y, T)
