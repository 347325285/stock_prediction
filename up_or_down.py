import numpy as np
import pandas as pd
import pandas_datareader.data as web
import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random


Yahoo1 = web.DataReader('600015.SS', data_source='yahoo', start='1/1/2017', end='2/14/2018')
Yahoo2 = web.DataReader('600030.SS', data_source='yahoo', start='1/1/2017', end='2/14/2018')
y1 = Yahoo1.iloc[:,4].as_matrix().tolist()
x1 = list(range(len(y1)))
plt.plot(x1,y1,'b', label=(('600015')))
y2 = Yahoo2.iloc[:,4].as_matrix().tolist()
x2 = list(range(len(y2)))
plt.plot(x2,y2,'r', label=(('600030')))
plt.legend()
plt.show()

y1 = Yahoo1.iloc[:,4].as_matrix().tolist()
y = Yahoo1.iloc[:,:].as_matrix()
Y1 = [1]
for i in range(len(y1)-1):
    if y1[i+1] >= y1[i]:
        Y1.append(1)
    else:
        Y1.append(0)

for i in range(y.shape[0]):
    y[i,5] *= 1e-8
feature_num = y.shape[1]
#################
hidden_dim = 80
iter_num = 2e+5
lambda1 = 1e-4
lr = 1e-4
given = 50
wanted = 1
###changeable###


def generator(Input, shape, minval=-1, maxval=1, active=tf.nn.sigmoid):
    weights = tf.get_variable(name='weights', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=minval, maxval=maxval))
    biases = tf.get_variable(name='biases', shape=[1, shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    if active == None:
        outputs = tf.matmul(Input, weights) + biases
    else:
        outputs = active(tf.matmul(Input, weights) + biases)
    return weights, biases, outputs


X = tf.placeholder(dtype=tf.float32, shape=[None, given*feature_num])
Y = tf.placeholder(dtype=tf.float32, shape=[None, wanted])
regularizer = tf.contrib.layers.l2_regularizer(lambda1)

with tf.variable_scope('Hidden'):
    weights, biases, hidden_outputs = generator(X, [given*feature_num, hidden_dim])
    tf.add_to_collection('Weights', weights)

with tf.variable_scope('Output'):
    weights, biases, outputs = generator(hidden_outputs, [hidden_dim, wanted], active=None)
    tf.add_to_collection('Weights', weights)

Weights = tf.get_collection('Weights')
predict = outputs
loss_pred = tf.reduce_mean(tf.square(outputs-Y))
loss_reg = tf.contrib.layers.apply_regularization(regularizer, weights_list=Weights)
loss = loss_pred + loss_reg
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    STEP = []
    ACCURACY = []
    for step in range(int(iter_num+1)):
        start = random.randint(0,len(x1)-(given+wanted))

        feed_dict = {
        X: np.array(y[start:(start+given),:]).reshape([1, -1]),
        Y: np.array(Y1[(start+given):(start+given+wanted)]).reshape([1, -1])
        }
        sess.run(train_op, feed_dict=feed_dict)

        if step % 5000 == 0:
            temp_p = []
            for i in range(len(x1)-(given+wanted)+1):
                start = i
                feed_dict = {
                X: np.array(y[start:(start+given),:]).reshape([1, -1]),
                Y: np.array(Y1[(start+given):(start+given+wanted)]).reshape([1, -1])
                }
                if sess.run(predict, feed_dict=feed_dict)[0,0] <= 0.5:
                    temp_p.append(0)
                else:
                    temp_p.append(1)
            k = 0
            for i in range(len(temp_p)):
                if temp_p[i] == Y1[given+i]:
                    k += 1
            accuracy = k/len(temp_p)
            STEP += [step//5000]
            ACCURACY += [accuracy]
            print('Step is {}, Accuracy is {:.2f}%'.format(step, accuracy*100))

print('ALL')
plt.plot(STEP, ACCURACY, 'r', label=(('ACCURACY')))
plt.legend()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    STEP = []
    ACCURACY = []
    for step in range(int(iter_num+1)):
        start = random.randint(0,len(x1)-(given+wanted)-20)

        feed_dict = {
        X: np.array(y[start:(start+given),:]).reshape([1, -1]),
        Y: np.array(Y1[(start+given):(start+given+wanted)]).reshape([1, -1])
        }
        sess.run(train_op, feed_dict=feed_dict)

        if step % 5000 == 0:
            temp_p = []
            for i in range(len(x1)-(given+wanted)+1):
                start = i
                feed_dict = {
                X: np.array(y[start:(start+given),:]).reshape([1, -1]),
                Y: np.array(Y1[(start+given):(start+given+wanted)]).reshape([1, -1])
                }
                if sess.run(predict, feed_dict=feed_dict)[0,0] <= 0.5:
                    temp_p.append(0)
                else:
                    temp_p.append(1)
            k = 0
            for i in range(20):
                if temp_p[len(x1)-given-20+i] == Y1[len(x1)-20+i]:
                    k += 1
            accuracy = k/20
            STEP += [step//5000]
            ACCURACY += [accuracy]
            print('Step is {}, Accuracy is {:.2f}%'.format(step, accuracy*100))

print('-20')
plt.plot(STEP, ACCURACY, 'r', label=(('ACCURACY')))
plt.legend()
plt.show()
