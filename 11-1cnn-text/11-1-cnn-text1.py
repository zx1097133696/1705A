# 文本卷积
import tensorflow as tf
import numpy as np
import re
from tensorflow.contrib import learn

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
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
batch_size = 64 #每批样本

dev_sample_percentage = 0.1 #测试集的比例
positive_data_file ='rt-polarity.pos'  #正面评价文本
negative_data_file ='rt-polarity.neg'  #负面评价文本
positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
positive_examples = [s.strip() for s in positive_examples]
print(positive_examples) #正面评价文本（临时测试用）
negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
negative_examples = [s.strip() for s in negative_examples]
print(negative_examples) #负面评价文本（临时测试用）

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

# tf.add_to_collection('x_dev',x_dev)
# tf.add_to_collection('y_dev',y_dev)
# tf.add_to_collection('x_train',x_train)
# tf.add_to_collection('y_train',y_train)

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

TB_SUMMARY_DIR = 'textdir1'
# 定义占位符
X = tf.placeholder(tf.int32, [None, max_document_length],name='X')
# 加入嵌入层
embedding_size = 8 # 词向量的维度
W = tf.Variable(tf.random_uniform([len(vocab_processor.vocabulary_), embedding_size], -1.0, 1.0),name='W')
embedded_chars = tf.nn.embedding_lookup(W, X)
X_img = tf.expand_dims(embedded_chars, -1) # 变成四维数据 [?, max_document_length, embedding_size, 1])
print(X_img) #（临时测试用）
Y = tf.placeholder(tf.float32, [None, 2],name='Y') # 独热编码

# 第1层卷积
W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01),name='W1')
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob=0.9)

# 第2层卷积
W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01),name='W2')
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob=0.9)

dim = L2.get_shape()[1].value * L2.get_shape()[2].value * L2.get_shape()[3].value
L2_flat = tf.reshape(L2,[-1, dim])

# 全连接层
W = tf.get_variable("W", shape=[dim, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]),name='b')
logits = tf.matmul(L2_flat, W) + b

# 代价或损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y),name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 优化器

tf.add_to_collection('train',optimizer)

tf.summary.scalar("loss", cost)
summary = tf.summary.merge_all()
global_step = 0

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='acc')

# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
writer = tf.summary.FileWriter(TB_SUMMARY_DIR, sess.graph)

saver = tf.train.Saver()
saver.save(sess,'./cnn_text/MyModel')

print('开始学习...')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(total / batch_size)  #计算总批次
    g_b = 0
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        s, c, _ = sess.run([summary, cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        writer.add_summary(s, global_step=global_step)
        global_step = global_step + 1
    acc = sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev})
    print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
print('学习完成')

# 测试模型检查准确率
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_dev, Y: y_dev}))
