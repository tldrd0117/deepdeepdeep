
import numpy as np
import matplotlib.pyplot as plt

TRAIN_NUM = 100


X1 = np.random.randn(TRAIN_NUM, 2) + np.array([0, 10])
Y1 = np.array([[1,0,0] for i in range(TRAIN_NUM)])

X2 = np.random.randn(TRAIN_NUM, 2) + np.array([5, 5])
Y2 = np.array([[0,1,0] for i in range(TRAIN_NUM)])

X3 = np.random.randn(TRAIN_NUM, 2) + np.array([10, 0])
Y3 = np.array([[0,0,1] for i in range(TRAIN_NUM)])


#print(X1)
#print(Y1)

plt.scatter(X1[:,0], X1[:,1])
plt.scatter(X2[:,0], X2[:,1])
plt.scatter(X3[:,0], X3[:,1])
plt.show()