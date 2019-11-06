# 文本分类RNN
import tensorflow as tf
import numpy as np
import re
from tensorflow.contrib import learn
from tensorflow.contrib.layers import fully_connected
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
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


#切分单词
x_text = positive_examples + negative_examples  #把所有正面和负面文本拼接起来
x_text = [clean_str(sent) for sent in x_text]   #去掉无效字符并切分单词
#生成标签
positive_labels = [[0, 1] for _ in positive_examples] # 独热编码，正面
negative_labels = [[1, 0] for _ in negative_examples] # 独热编码，负面
y_data = np.concatenate([positive_labels, negative_labels], 0)  #连接所有标签
print(x_text) #经过处理的文本（临时测试用）
print(y_data) #标签（临时测试用）

#建立词汇表
max_document_length = max([len(x.split(" ")) for x in x_text])  #返回一个句子中最大的单词数
print('一个句子最大的单词数：', max_document_length)  #（临时测试用）
#把所有的单词重新编码
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print('单词总数：', len(vocab_processor.vocabulary_)) #（临时测试用）
print('句子的编码', x) #（临时测试用）

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_data)))
x_shuffled = x[shuffle_indices]
y_shuffled = y_data[shuffle_indices]

#切分训练集和测试集
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y_data)))
x_train, x_dev = np.split(x_shuffled, [dev_sample_index,])
y_train, y_dev = np.split(y_shuffled, [dev_sample_index,])

total = x_train.shape[0]
sequence_length = x_train.shape[1]
print('训练集', x_train.shape, '(句子数,每个句子的最大单词数)') #（临时测试用）
print('测试集：', x_dev.shape) #（临时测试用）

g_b=0
# 自己实现next_batch函数，每次返回一批数据
def next_batch(size):
    global g_b
    xb = x_train[g_b:g_b+size]
    yb = y_train[g_b:g_b+size]
    g_b = g_b + size
    return xb,yb

n_neurons = 128 #隐藏状态，神经元个数
n_outputs = 2 #输出2分类
n_layers = 2 #层数

embedding_size = n_neurons # 词向量的维度
batch_size = 64 #每批样本
n_steps = max_document_length #时间步数（序列长度）
n_inputs = embedding_size #输入数据长度

# 定义占位符
X = tf.placeholder(tf.int32, [None, max_document_length],name='X')
Y = tf.placeholder(tf.int32, [None, 2],name='Y') # 独热编码
# 加入嵌入层
W = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), embedding_size], -1.0, 1.0),name='W')
X_data = tf.nn.embedding_lookup(W, X)  #(?, 56, 128)
print(X_data) #（临时测试用）

#模型
# cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
# cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons) for layer in range(n_layers)]
cells = [tf.contrib.rnn.LSTMCell(num_units=n_neurons) for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X_data, dtype=tf.float32) #outputs(?, 56, 128)
print(outputs.shape)
# top_layer_h_state = states[-1][1] #最顶层隐藏层状态（最后）  (?,128)
# logits = tf.layers.dense(outputs[:,-1], n_outputs, name="softmax") #用最后一个Cell的输出
logits = fully_connected(outputs[:,-1], n_outputs, activation_fn=None) #用最后一个Cell的输出

# 代价或损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y),name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 优化器

tf.add_to_collection('train',optimizer)

# tf.summary.scalar("loss", cost)
# summary = tf.summary.merge_all()
# global_step = 0

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='acc')

# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
# writer = tf.summary.FileWriter(TB_SUMMARY_DIR, sess.graph)

saver = tf.train.Saver()
saver.save(sess,'./rnn_text/MyModel')

print('开始学习...')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(total / batch_size)  #计算总批次
    g_b = 0
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        # s, c, _ = sess.run([summary, cost, optimizer], feed_dict=feed_dict)
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        # writer.add_summary(s, global_step=global_step)
        # global_step = global_step + 1
    acc = sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev})
    print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
print('学习完成')

# 测试模型检查准确率
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev}))

'''
Epoch: 1 cost = 0.6905622550305104 acc= 0.6163227
Epoch: 2 cost = 0.5400942524007504 acc= 0.72232646
Epoch: 3 cost = 0.35755642148472305 acc= 0.7448405
Epoch: temp1 cost = 0.2213559322209166 acc= 0.750469
Epoch: 5 cost = 0.14684994823780637 acc= 0.74108815
Epoch: 6 cost = 0.09073889089885778 acc= 0.74859285
Epoch: 7 cost = 0.07446735336964061 acc= 0.750469
Epoch: 8 cost = 0.049180956293309985 acc= 0.7307692
Epoch: 9 cost = 0.030468408543391504 acc= 0.7429643
Epoch: 10 cost = 0.02506378301084916 acc= 0.74108815
学习完成
Accuracy: 0.74108815(GRUCell)
0.74202627(LSTM)
0.48311445 BasicRNNCell
'''
