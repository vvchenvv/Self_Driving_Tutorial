至此我们知道，我们通过权重和偏置（weights and biases）去计算node的输出，然后通过激活函数去将输出归类。神经网络通过输入经过标记的数据进行训练，不断修改权重和偏置提高输出的精度。当前有多种办法去衡量一个神经网络的输出精度，这些办法都以网络产生尽可能接近已知正确值的值的能力为核心。人们使用不同的名称表示精确度测量，通常将其称为损失（Lost）或成本（Cost）。

# 均方误差MSE
MSE：mean squared error 

![image](http://note.youdao.com/favicon.ico)

- w：神经网络中所有权重的集合
- b：所有偏置的集合
- m：所有训练数据的数量
- a：对应y（x）的真实值

Cost： C依赖于y（x）和a之间的差，如果所有的数据产生的y（x）和对应的a都没有误差，则整个神经网络的误差为0.这是最理想的状态，学习的过程就是将Cost不断逼近0的过程。

# MSE的代码实现


```
class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        # TODO: your code here
        diff = y-a
        self.value = np.mean(np.square(diff))
        pass
```

这里面使用reshape（）是防止矩阵在传播过程中因矩阵不一致问题导致的计算错误。

更多代码请关注
[我的Github](http://note.youdao.com/)

[我的网站](http://weiweizhao.com/category/ai/)


