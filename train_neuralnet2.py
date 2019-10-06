# In[1]: train
import sys, os
import numpy as np
from three_layer_net import ThreeLayerNet

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

# plt.scatter(X1[:,0], X1[:,1])
# plt.scatter(X2[:,0], X2[:,1])
# plt.scatter(X3[:,0], X3[:,1])
# plt.show()

x_train = np.vstack([X1, X2, X3])
t_train = np.vstack([Y1, Y2, Y3])

print(x_train, t_train)
print(len(x_train), len(t_train))
network = ThreeLayerNet(input_size=2, hidden_size=10, hidden_size2=10, output_size=3)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 1
learning_rate = 0.000001

train_loss_list = []
train_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1','W2', 'b2','W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]
        # print(key, network.params[key])
        # print(key, grad[key])
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
    # if i % (iters_num/20) == 0:
    print('loss', loss)
        # print(train_acc)
# print(train_loss_list)
# print(train_acc_list)
print(network.params)
print(grad)
# In[2]: run
my = np.array([[58, 30]])
print(my, network.predict(my))

my2 = np.array([[158, 130]])
print(my2, network.predict(my2))



#%%
