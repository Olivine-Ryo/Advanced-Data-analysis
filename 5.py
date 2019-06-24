import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
np.random.seed(1)


def generate_train_datax(data):
    x=np.zeros((256,0))
    for j in range(0,10):
        x = np.concatenate([x, data['X'][:, :, j]], axis=1)     
    x = np.transpose(x, (1, 0))
    return x

def generate_train_datay(data,i):
    size=data['X'][:, :, i].shape[1]
    y=np.zeros(0)
    for j in range(0,10):
        if i == j:
           y = np.concatenate([y, np.ones(size)])
        else:
           y = np.concatenate([y, -np.ones(size)])            
    return y

def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))
    
def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10.).T.dot(theta)

def build_confusion_matrix(train_data, data, theta):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    #正解データ0~9に対して確認
    for i in tqdm(range(10)):
        test_data = np.transpose(data[:, :, i], (1, 0))
        prediction = predict(train_data, test_data, theta)
        pred_num=np.argmax(prediction,axis=1)
        #予測値0~9に対して確認
        for j in range(10):
            confusion_matrix[i][j]=np.sum(np.where(pred_num == j, 1, 0))
    return confusion_matrix

def main(x,data,i):
    y = generate_train_datay(data,i)
    design_mat = build_design_mat(x, x, 10.)
    theta = optimize_param(design_mat, y, 1.)
    
    return theta

data = loadmat('digit.mat')
data['X'] = np.delete(data['X'],np.arange(150,500),1)
x=generate_train_datax(data)

print("Generating optimal parameters")
THETA=[]
for i in tqdm(range(0,10)):
    THETA.append(main(x,data,i))
THETA=np.array(THETA).T

print("Generating confusion matrix")
confusion_matrix = build_confusion_matrix(x, data['T'], THETA)
print('confusion matrix:')
print(confusion_matrix)
print('accuracy: %.2f' % np.trace(confusion_matrix)/np.sum(confusion_matrix))
