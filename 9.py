import numpy as np
import matplotlib


import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y

def generate_phi(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))
    
def k_means(x, k):
    phi=generate_phi(x,x,1)
    N=phi.shape[0]
    W=np.zeros([0,N])
    for i in range(0,N):
        W=np.vstack([W,np.where(phi[i]>sorted(phi[i])[-(k+1)],1,0)])
        
    for i in range(0,N):
        for j in range(0,N):
            W[i][j]=max(W[i][j],W[j][i])
    return W
def lrls(x, y, h=1., l=1., nu=1.):
    """

    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """
    tilt_phi=generate_phi(x, x[y != 0],1.)
    phi=generate_phi(x, x,1.)
    W=k_means(x,7)
    D=np.diag(np.sum(W,axis=1))
    L=D-W
    
    A=np.dot(tilt_phi, tilt_phi.T) + l*np.eye(len(x)) +\
    2 * nu * np.dot(phi.T, np.dot(L, phi)) 
    B=np.dot(tilt_phi, y[y!=0])
    theta=np.linalg.solve(A,B)
    #
    # Implement this function
    #

    return theta


def visualize(x, y, theta, h=1.):
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(-20., 20.)
    plt.ylim(-20., 20.)
    grid_size = 100
    grid = np.linspace(-20., 20., grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    k = np.exp(-np.sum((x.astype(np.float32)[:, None] - mesh_grid.astype(
        np.float32)[None]) ** 2, axis=2).astype(np.float64) / (2 * h ** 2))
    plt.contourf(X, Y, np.reshape(np.sign(k.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='$.$', c='black')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$X$', c='red')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
    plt.savefig('lecture9-h1.png')
    return k

x, y = generate_data(n=200)
theta = lrls(x, y, h=1.)
k=visualize(x, y, theta)
