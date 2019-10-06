# In[1]: train
import sys, os
import numpy as np
from three_layer_net import ThreeLayerNet
import matplotlib.pyplot as plt

import requests

res = requests.get('https://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt')
lines = res.text.splitlines()[36:-1]
x_train = np.array(list(map(lambda x : list(map(lambda d : float(d),list(filter(lambda text: len(text)>0, x.split(' ')))[2:4])),lines)))
t_train = np.array(list(map(lambda x :[ float(list(filter(lambda text: len(text)>0, x.split(' ')))[4])],lines)))
print(x_train, t_train)
print(len(x_train), len(t_train))
network = ThreeLayerNet(input_size=2, hidden_size=1, hidden_size2=1, output_size=1)

iters_num = 50000
train_size = x_train.shape[0]
batch_size = 5
learning_rate = 0.0001

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
    if i % (iters_num/20) == 0:
        print('loss', loss)
        # print(train_acc)
# print(train_loss_list)
# print(train_acc_list)
print(network.params)
print(grad)
print('loss', loss)
# In[2]: run
my = np.array([[58, 30]])
print(my, network.predict(my))

my2 = np.array([[100, 50]])
print(my2, network.predict(my2))



#%%
