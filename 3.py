from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

%matplotlib inline
import matplotlib.pyplot as plt

np.random.seed(10) 

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# calculate design matrix
h = 0.7
k = calc_design_matrix(x, x, h)

# solve the least square problem
l =0.1
O=np.zeros(sample_size).T
I=np.identity(len(k))


theta=np.random.rand(sample_size)
z=np.random.rand(sample_size)
u=np.random.rand(sample_size)


Theta=[]
Z=[]
U=[]
Time=200

temp1=k.T.dot(k) + I
temp2=k.T.dot(y)
for i in range(0,Time):
    
    Theta.append(theta)
    Z.append(z)
    U.append(u)
    
    theta = np.linalg.solve(temp1,temp2 + z - u)
    z=np.maximum(O,-l+theta+u)+np.minimum(O,l+theta+u)
    u=u+theta-z

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, h)
prediction = K.dot(theta)

# visualization
plt.figure(figsize=(10, 5))
plt.clf()
plt.grid(color='gray')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)

plt.figure(figsize=(20, 10), dpi=50)
plt.grid(color='gray')
plt.xlabel("The number of updates")
plt.ylabel("theta")
plt.plot(np.array(Theta)[:,])