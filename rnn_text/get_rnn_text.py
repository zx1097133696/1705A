#! -*- coding:utf-8 -*-
#! @time     : 2019/11/06 20:31:30
#! @Author   : zhangxu
#! @File     : get_rnn_text.py
#! @Software :PyCharm
import tensorflow as tf
import numpy as np
import re
from tensorflow.contrib import learn
from tensorflow.contrib.layers import fully_connected

tf.set_random_seed(777)  #设置随机种子

def clean_str(string): #去掉文本中的无效字符并切分单词
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

learning_rate = 0.001  #学习率
training_epochs = 10 #训练总周期

dev_sample_percentage = 0.1 #测试集的比例
positive_data_file ='rt-polarity.pos'  #正面评价文本
negative_data_file ='rt-polarity.neg'  #负面评价文本
positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
positive_examples = [s.strip() for s in positive_examples]
print(positive_examples) #正面评价文本（临时测试用）
negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
negative_examples = [s.strip() for s in negative_examples]
print(negative_examples) #负面评价文本（临时测试用）l

#生成标签
positive_labels = [[0, 1] for _ in positive_examples] # 独热编码，正面
negative_labels = [[1, 0] for _ in negative_examples] # 独热编码，负面
y_data = np.concatenate([positive_labels, negative_labels], 0)  #连接所有标签

#切分单词
x_text = positive_examples + negative_examples  #把所有正面和负面文本拼接起来
x_text = [clean_str(sent) for sent in x_text]   #去掉无效字符并切分单词

#建立词汇表
max_document_length = max([len(x.split(" ")) for x in x_text])  #返回一个句子中最大的单词数

#把所有的单词重新编码
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_data)))
x_shuffled = x[shuffle_indices]
y_shuffled = y_data[shuffle_indices]

#切分训练集和测试集
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y_data)))
x_train, x_dev = np.split(x_shuffled, [dev_sample_index,])
y_train, y_dev = np.split(y_shuffled, [dev_sample_index,])

total = x_train.shape[0]

g_b=0
# 自己实现next_batch函数，每次返回一批数据
def next_batch(size):
    global g_b
    xb = x_train[g_b:g_b+size]
    yb = y_train[g_b:g_b+size]
    g_b = g_b + size
    return xb,yb

batch_size = 64 #每批样本

with tf.Session() as sess:
    new_parth = tf.train.import_meta_graph('./rnn_text/MyModel.meta')
    new_parth.restore(sess,tf.train.latest_checkpoint('./rnn_text'))

    X = sess.graph.get_tensor_by_name('X:0')
    Y = sess.graph.get_tensor_by_name('Y:0')
    cost = sess.graph.get_tensor_by_name('cost:0')
    train = sess.graph.get_collection('train')
    acc = sess.graph.get_tensor_by_name('acc:0')

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(total / batch_size)  # 计算总批次
        g_b = 0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, train], feed_dict=feed_dict)
            avg_cost += c / total_batch
        acc_ = sess.run(acc, feed_dict={X: x_dev, Y: y_dev})
        print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc_)
    print('学习完成')

    # 测试模型检查准确率
    print('Accuracy:', sess.run(acc, feed_dict={X: x_dev, Y: y_dev}))
