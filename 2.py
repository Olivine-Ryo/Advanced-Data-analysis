
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

%matplotlib inline
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def karnel(x,c,h):
    return np.exp()


# create sample
sample_size = 30
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)
N=len(x)-1
sum=0


# set parameters
H=[0.1,0.5,1]
L=[0.01,0.05,0.1]

fig, axes= plt.subplots(len(H),len(L),figsize=(12,12))

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=1000)

for m in range(0,len(H)):
    for n in range(0,len(L)):
        sum=0
        for i in range(0,N):

            x_test=np.delete(x,i)
            y_test=np.delete(y,i)

            # calculate design matrix
            h = H[m]
            k = calc_design_matrix(x_test, x_test, H[m])

            # solve the least square problem
            l = L[n]
            theta = np.linalg.solve(
                k.T.dot(k) + l * np.identity(len(k)),
                k.T.dot(y_test[:, None]))

            # create data to visualize the prediction
            #X = np.linspace(start=xmin, stop=xmax, num=100)
            K = calc_design_matrix(x_test, X, H[m])
            prediction = K.dot(theta)

            predict=calc_design_matrix(x[i],x_test, H[m]).T.dot(theta)
            real=y_test[i]

            sum = (predict-real)**2 + sum          
            
        # visualization
        ax=axes[m,n]
        ax.scatter(x_test, y_test, c='green', marker='o')
        ax.plot(X, prediction)
        ax.set_title("h=%0.1f ,lambda=%0.2f, sum=%0.3f" % (h, l, sum))



