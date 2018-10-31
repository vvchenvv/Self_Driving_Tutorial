通过上一章节我们得到了每个输入对cost的梯度值，也就是forward_and_backward()函数的返回值。通过这个函数，我们的神经网络已经具备了学习的条件，为了使它开始学习，我们引入随机梯度下降方法。
<!--more-->
# 随机梯度下降
随机梯度下降（SGD）是梯度下降的一个版本，每一个回合都从所有数据集中随机选取一部分数据集输入到神经网络中。理想情况下我们应该把全部的数据集一次输入到神经网络中，但是在实际中由于内存的限制，我们无法实现这样的训练方式。SGD是梯度下降的一个近似值，当输入神经网络的批次越多，神经网络预测的结果越好。

以下是简单的SGD步骤：
1. 从总数据中随机选取一批数据
2. 使用步骤1的数据运行网络的forward和backward函数，计算梯度
3. 运行梯度下降更新方法
4. 重复步骤1-3

我们会发现网络的输出的cost值会有明显下降的趋势。到目前为止，我们的MiniFlow可以运行到步骤2，我们将实现剩余步骤。
复习一下梯度下降的更新公式：

> x=x−α∗(∂cost/∂x)

我们使用sklearn自带的波士顿房价数据集来进行训练。关于这个数据集的更多信息请关注链接[sklearn提供的自带的数据集](https://www.cnblogs.com/nolonely/p/6980160.html)。

# 代码实现

```
def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial  #参考上面的梯度下降更新公式
```

SGD实现方式：

```
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value
```
上面用到了sklearn自带的resample函数，作用是随机选取指定数量的数据集。更详细用法请参考官方文档：
[sklearn.utils.resample](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)
测试结果：

![image](http://note.youdao.com/favicon.ico)


完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/tree/master/Class1/29_Stochastic_Gradient_Descent)

更多文章请关注我的个人网站：
[link](http://weiweizhao.com/category/ai/)

