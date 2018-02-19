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
#################
hidden_dim = 25
iter_num = 2e+5
lambda1 = 1e-4
lr = 1e-4
given = 50
wanted = 10
###changeable###


def generator(Input, shape, minval=-1, maxval=1, active=tf.nn.sigmoid):
    weights = tf.get_variable(name='weights', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=minval, maxval=maxval))
    biases = tf.get_variable(name='biases', shape=[1, shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    if active == None:
        outputs = tf.matmul(Input, weights) + biases
    else:
        outputs = active(tf.matmul(Input, weights) + biases)
    return weights, biases, outputs


X = tf.placeholder(dtype=tf.float32, shape=[None, given])
Y = tf.placeholder(dtype=tf.float32, shape=[None, wanted])
regularizer = tf.contrib.layers.l2_regularizer(lambda1)

with tf.variable_scope('Hidden'):
    weights, biases, hidden_outputs = generator(X, [given, hidden_dim])
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

    for step in range(int(iter_num+1)):
        start = random.randint(0,len(x1)-(given+wanted))
        feed_dict = {
        X: np.array(y1[start:(start+given)]).reshape([-1, given]),
        Y: np.array(y1[(start+given):(start+given+wanted)]).reshape([-1, wanted])
        }
        sess.run(train_op, feed_dict=feed_dict)
    print('Finished')
    temp_m = np.zeros([len(x1)-(given+wanted)+1,len(x1)])

    for i in range(len(x1)-(given+wanted)+1):
        start = i
        feed_dict = {
        X: np.array(y1[start:(start+given)]).reshape([-1, given]),
        Y: np.array(y1[(start+given):(start+given+wanted)]).reshape([-1, wanted])
        }

        temp_m[[i],:] = [0]*(i+given) + list(np.squeeze(sess.run(predict, feed_dict=feed_dict))) + [0]*(len(x1)-(given+wanted)-i)
    pred_temp = list(np.sum(temp_m, axis=0))

    pred = []
    for i in range(len(pred_temp)):
        j = i+1
        if j <= given:
            pred.append(9)
        elif j > given and j < given+wanted:
            pred.append(pred_temp[i]/(j-given))
        elif j >= given+wanted and j <= len(x1)-wanted:
            pred.append(pred_temp[i]/wanted)
        else:
            pred.append(pred_temp[i]/(len(x1)-i))

print('ALL')
y1 = Yahoo1.iloc[:,4].as_matrix().tolist()
x1 = list(range(len(y1)))
plt.plot(x1,y1,'b', label=(('TRUE')))
y2 = Yahoo2.iloc[:,4].as_matrix().tolist()
plt.plot(x1[given:],pred[given:],'r', label=(('PREDICT')))
plt.legend()
plt.show()

with tf.Session() as sess:
    sess.run(init)

    for step in range(int(iter_num+1)):
        start = random.randint(0,len(x1)-(given+wanted)-20)
        feed_dict = {
        X: np.array(y1[start:(start+given)]).reshape([-1, given]),
        Y: np.array(y1[(start+given):(start+given+wanted)]).reshape([-1, wanted])
        }
        sess.run(train_op, feed_dict=feed_dict)
    print('Finished')
    temp_m = np.zeros([len(x1)-(given+wanted)+1,len(x1)])

    for i in range(len(x1)-(given+wanted)+1):
        start = i
        feed_dict = {
        X: np.array(y1[start:(start+given)]).reshape([-1, given]),
        Y: np.array(y1[(start+given):(start+given+wanted)]).reshape([-1, wanted])
        }

        temp_m[[i],:] = [0]*(i+given) + list(np.squeeze(sess.run(predict, feed_dict=feed_dict))) + [0]*(len(x1)-(given+wanted)-i)
    pred_temp = list(np.sum(temp_m, axis=0))

    pred = []
    for i in range(len(pred_temp)):
        j = i+1
        if j <= given:
            pred.append(9)
        elif j > given and j < given+wanted:
            pred.append(pred_temp[i]/(j-given))
        elif j >= given+wanted and j <= len(x1)-wanted:
            pred.append(pred_temp[i]/wanted)
        else:
            pred.append(pred_temp[i]/(len(x1)-i))

print('-20')
y1 = Yahoo1.iloc[:,4].as_matrix().tolist()
x1 = list(range(len(y1)))
plt.plot(x1[250:],y1[250:],'b', label=(('TRUE')))
y2 = Yahoo2.iloc[:,4].as_matrix().tolist()
plt.plot(x1[250:],pred[250:],'r', label=(('PREDICT')))
plt.legend()
plt.show()
