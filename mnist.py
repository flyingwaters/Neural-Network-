from tensorflow.examples.tutorials.mnist import input_data
#import function from module
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#作为下载的 mnist的脚本，使用one_hot = True 独热编码
#######################################################################3
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
import tensorflow as tf
sess = tf.InteractiveSession()
#这个命令将这个session 注册为默认的session 之后的运算也跑在这个session里
x = tf.placeholder(tf.float32,[None,784])
#None 代表不限条数
#Variable是用来存储模型参数的，不用于存储数据的tensor 一旦使用掉就会消失
#Variable是持久的，每轮迭代中被更新。复杂的网络的初始化比较重要
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w)+b)
#tf.nn 是大量神经网络的组件 tf.matmul()是TensorFlow 中的矩阵乘法函数
#tensorflow 自动实现forward 和backward 的内容。只要定义好loss
#对于多分类问题，cross-entroy作为loss function。Cross-entropy最早出自
#Cross-entroy最早出自信息论中的信息熵（与压缩比率有关），然后被用到
#通信，纠错码，博弈论，机器学习等H = - 求和有ilog（yi）
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# axis 的过去式 reduction_indices
#定义一个优化算法，我们就可以开始训练了
#SGD（Stochastic Gradient Descent）。定义好算法后。Tensorflow就可以根据我们定义的整个计算图
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
#全局参数初始化器，并执行它的run方法

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
# 训练简单完成
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
# 1:定义算法公式，也就是神经网络forward时的计算
# 2：定义loss，选定优化器，并指定优化器优化loss
# 3：迭代地对数据进行训练
# 4：在测试集或验证集上对准确率进行评测
#这是一个计算图，我们定义cross——entropy,train_step,accuracy都是计算图
#中的节点，而并不是数据结果，
# 我们可以通过调用run方法执行这些节点或者说运算操作来获取结果
#LeCun的LeNet5 在20世纪90年代提出99%的准确率，可以说时领先时代的大突破
#2006 Hinton 逐层预训练来初始化权重的方法及利用多层RBM堆叠的神经网络DBN
#2012 ALexNet获得ImageNet ILSVRC比赛的第一名
#Softmax Regression 加入一个隐含层变成一个正统神经网络后，DropOut，Agrad，
#ReLU等技术准确率可以达到98%.引入卷积层和池化层后，也可以达到99%.
#目前基于卷积神经网络的state-of-art 的方法已经可以达到99.8%
