import tensorflow as tf
tf.set_random_seed(777) #设置随机种子
#定义数据集
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
#定义占位符
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
#权重和偏置
W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#预测模型
hypothesis = tf.matmul(X, W) + b
#代价或损失函数
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(2001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:# 显示损失值收敛情况
        print(step, cost_val)
#验证或预测
print(sess.run(hypothesis, feed_dict={X: [[5,6,7]]}))
