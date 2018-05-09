#加上隐含层
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

#给隐含层参数设置Variable并进行初始化，
#这里in_units是输入节点数，h1_units即隐含层的输出节点数设为300
in_units = 784
hl_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units,hl_units],stddev=0.1))
#tf.truncated_normal(shape,mean,stddev):
#产生的正太分布的值如果与均值的差值大于两倍的标准差，就重新生成
b1 = tf.Variable(tf.zeros([hl_units]))
W2 = tf.Variable(tf.zeros([hl_units,10]))
b2 = tf.Variable(tf.zeros([10]))

#输入x 的placeholder
#keep_prob 的 placeholder
x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
# tf.nn.dropout(x,keep_prob)#去除掉一些 hidden1的输出
y = tf.nn.softmax((tf.matmul(hidden1_drop,W2)+b2))
#1：定义算法公式
#2：定义loss ，选择优化器
#3：训练
#4：评测
y_ = tf.placeholder(tf.float32,[None,10])
#训练集标签
cross_entropy = tf.reduce_sum(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#优化器，学习率明确一个数值
#加入了keep_prob 作为计算图的输入，并且训练时设为0.75
#更多的训练迭代来优化模型参数以达到一个比较好的效果
#采用了3000个batch
tf.global_variables_initializer().run()
#初始化Variables
for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    # return self._images[start:end], self._labels[start:end]
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
    #run 最后一个node操作
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#tf.argmax,返回每一列的最大值的的索引号,若矩阵返回向量，若向量返回一个值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#tf.cast(x,dtype)---->转换为一个新类型
#tf.reduce_mean(x,axis=None)
#没参数求平均数
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
