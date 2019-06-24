import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import dot
np.random.seed(1)  # set the random seed for reproducibility


def data_generate(n=50):
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]

def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))
    
def online(x, y, l):
    theta = np.ones(3)
    sigma = np.ones(3)
    prev_w = np.zeros(3)
    gamma=0.0001
    for i in range(20):     
        j=np.random.randint(0,len(y))

        xi, yi = x[j], y[j]
        
        beta= np.dot(xi,np.dot(sigma,xi))+gamma
        print(beta)
        theta = theta + (yi*max(0,1-yi*np.dot(theta,xi))) / beta * sigma * xi
        sigma = sigma - np.dot(sigma, np.dot(xi ,np.dot(xi, sigma))) / beta
        
        #if np.linalg.norm(w - prev_w) < 1e-3:
        #    break
        #prev_w = w.copy()
        '''
        tmp = np.dot(xi[:, np.newaxis], xi[:, np.newaxis].T)
        gamma = 0.0001   #小さめ
        beta = np.dot(np.dot(x[:, np.newaxis].T,  sigma) , x[:, np.newaxis])  + gamma
        sigma = sigma - np.dot(np.dot(sigma, tmp), sigma) / beta   #シグマ更新
        theta = theta + xi * np.max(1 - np.dot(theta[:, np.newaxis].T, x[i]) * t[i], 0) * np.dot(sigma, x[i]) / beta  #ミュー更新
        theta = theta.flatten()
        '''
    return theta

def f(x):   
    y = - theta[0] / theta[1] * x - theta[2] / theta[1]
    return y

x,y=data_generate()

#theta=online(x,y,1)
theta = np.random.normal(loc=0, scale=0.1, size=3)
sigma = np.random.normal(loc=0, scale=0.1, size=(3,3))
gamma=0.01
prev_theta = theta.copy()
for i in range(1000):     
    j=np.random.randint(0,len(y))
    xi, yi = x[j], y[j]
    
    tmp = np.dot(xi[:, np.newaxis],xi[:, np.newaxis].T)
    gamma = 0.0001
    beta = dot(dot(xi[:, np.newaxis].T,  sigma) , xi[:, np.newaxis])  + gamma
    sigma = sigma - dot(dot(sigma, tmp), sigma) / beta
    theta = theta + yi * np.max(1 - dot(theta[:, np.newaxis].T, xi) * yi, 0) * dot(sigma, xi) / beta
    theta = theta.flatten()
    if np.linalg.norm(theta - prev_theta) < 1e-6:
        print(i)
        break
    prev_theta = theta.copy()
    
    
print(theta)

x_a=np.zeros((0,3))
x_b=np.zeros((0,3))
i=0
for i in range(0,len(y)):
    if y[i] == 1:
        x_a=np.vstack((x_a,x[i]))
    elif y[i] == -1:
        x_b=np.vstack((x_b,x[i]))
        
plt.figure(figsize=(10,5))        
plt.xlim(-20,0)
plt.ylim(-5,5)
plt.scatter(x_a[:,0],x_a[:,1],color='r')
plt.scatter(x_b[:,0],x_b[:,1],color='b')
x_plt=np.arange(-20,0,0.5)
plt.plot(x_plt, f(x_plt))
plt.show()