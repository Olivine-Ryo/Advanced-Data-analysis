import numpy as np

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y

def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        (x1[:, None] - x2[ None]) ** 2 / (2 * bandwidth ** 2))

def calc(y,phi,l):
    size=phi.shape[0]
    
    y0=np.where(y==0,1,0)
    y1=np.where(y==1,1,0)
    y2=np.where(y==2,1,0)
    theta0=np.linalg.solve(phi.T.dot(phi)+l*np.identity(size),phi.T.dot(y0))
    theta1=np.linalg.solve(phi.T.dot(phi)+l*np.identity(size),phi.T.dot(y1))
    theta2=np.linalg.solve(phi.T.dot(phi)+l*np.identity(size),phi.T.dot(y2))
    return theta0,theta1,theta2

def visualize(x, y, theta,h):
    X = np.linspace(-5., 5., num=100)

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-1, 1)
    colors=['blue','red','green']
    q=[0,0,0]
    for i in range(0,3):
        q[i] = np.sum(theta[i][y==i]*np.exp(-(X[:,None]-x[y==i][None])**2/(2*h**2)),axis=1)
    
    sum_l=np.where(q[0]<0,0,q[0])+np.where(q[1]<0,0,q[1])+np.where(q[2]<0,0,q[2])
    
    q_cal=[0,0,0]
    for i in range(0,3):
        q_cal[i]=np.where(q[i]<0,0,q[i])/sum_l
        plt.plot(X, q_cal[i], c=colors[i],label='q(y=%d|x)' % int(i+1))


    plt.title('L2 norm = %.1f' % lam)
    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    
h=1
x, y = generate_data(sample_size=90, n_class=3)
phi=build_design_mat(x,x,h)
for lam in np.arange(0.8,1.3,0.1):
    theta = calc(y,phi,lam)
    visualize(x,y,theta,h)

