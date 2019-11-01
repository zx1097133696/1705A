#! -*- coding:utf-8 -*-
#! @time     : 2019/10/25 15:24:36
#! @Author   : zhangxu
#! @File     : 12_1_hello.py
#! @Software :PyCharm
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers

# 建立字典
idx2char = ['h', 'i', 'e', 'l', 'o']
#            0    1    2    3    4
x_data = [0,1,0,2,3,3] #hihell
x_one_hot = np.eye(5)[x_data].reshape(1,-1,5)
y_data = [[1,0,2,3,3,4]] #ihello

#设置参数
learning_rate = 0.1 #学习率
sequence_size = 6 #序列长度
hidden_size = 8 #隐藏神经元个数
classes_num = 5 #总类别数
input_dim = 5 #独热长度
batch_size = 1 #批次大小

#占位符
X = tf.placeholder(tf.float32,shape=[None,sequence_size,classes_num]) #?*6*5
Y = tf.placeholder(tf.int32,shape=[None,sequence_size]) #?*6

#创建LSTM单元
cell = rnn.LSTMCell(num_units=hidden_size,state_is_tuple=True)
initial_start = cell.zero_state(batch_size,dtype=tf.float32)
outputs,_ = dynamic_rnn(cell,inputs=X,initial_state=initial_start,dtype=tf.float32)

#全连接
x_fc = tf.reshape(outputs,shape=[-1,hidden_size])
fc = layers.fully_connected(inputs=x_fc,num_outputs=classes_num,activation_fn=None)

#序列损失
outputs= tf.reshape(fc,shape=[batch_size,sequence_size,classes_num])
weigth = tf.ones(shape=[batch_size,sequence_size])
loss = tf.reduce_mean(seq2seq.sequence_loss(logits=outputs,targets=Y,weights=weigth))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#准确率
prediction = tf.argmax(outputs,axis=2)
acc_prediction = tf.equal(prediction,y_data)
accuracy = tf.reduce_mean(tf.cast(acc_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l,_,a = sess.run([loss,train,accuracy],feed_dict={X:x_one_hot,Y:y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i,'Loss:',l,'Prediction:',result,'Y True',y_data,a)
        print(result_str,''.join(result_str))
        if a>=1.0:
            break

