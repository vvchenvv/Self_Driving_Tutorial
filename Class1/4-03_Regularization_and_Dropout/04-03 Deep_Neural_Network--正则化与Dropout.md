当我们的训练数据集比较小时候经常会出现训练集准确率很高接近100%，但测试集准确率却很差的问题，这是过拟合（over fitting）现象。解决过拟合现象经常使用正则化（Regularization）与Dropout。

# 正则化Regularization
深度学习的模型只有在有大量的训练数据时才会有明显效果，这是近年来随着大数据兴起后深度学习模型才流行起来的原因。大量数据也带来一个问题就是如何有效快速的利用大量数据进行训练，这就要依赖于正则化。正则化是在神经网络上增加一定的限制，用以间接减少自由变量的参数数量。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/16%20-%20Regularization.mp4_000023.712.jpg)

但同时不能增加模型的复杂度，不增加优化的难度。正则化分为L1正则化与L2正则化，具体介绍可以参考一些优秀的博客：[深度学习中的正则化](http://www.imooc.com/article/69484)
我们来看一下L2正则化：
它的核心思想是在Loss函数中加入一个额外的项以削减大权重的影响。做法是把权重矩阵的L2范数乘以一个小常数得到这个加入到Loss函数中的值。小常数β就是我们新引入的超参。
 L2范数的定义其实是一个数学概念，其定义如下：
 
 ![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/16%20-%20Regularization.mp4_000038.906.jpg)
 
 欧式距离就是一种L2范数，表示向量元素的平方和再开方。
 更详细的介绍可以参考优秀博客：[一文搞懂深度学习中的L2范数](https://blog.csdn.net/u010725283/article/details/79212762)
 
 ![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/L2%E8%8C%83%E5%BC%8F.png)
 
由于我们只是在Loss的计算中加入了一项，并没有改变我们的神经网络框架，因此神经网络的复杂度没有发生改变。使用L2范数的一个好处是L2范数的求导是我们的权重W矩阵本身。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/17%20-%20Regularization%20Quiz.mp4_000020.080.jpg)

# Dropout
Dropout的原理听起来有些疯狂。假设有两层神经元连接，从前一层到后一层的值叫激活值Activations。在训练神经网络的过程中随机选择一半的Activations设置为0，这个选择的过程完全随机。相当于将流过神经网络的一半数据直接忽略掉。每一回合的训练都重复这个过程，随机选取一半的数丢弃掉。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/%E5%85%A8%E7%BD%91%E7%BB%9C%E4%B8%8EDropout%E7%BD%91%E7%BB%9C.jpg)

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/18%20-%20Dropout.mp4_000018.948.jpg)

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/18%20-%20Dropout.mp4_000027.391.jpg)

Dropout的优点是我们的神经网络再也不能依赖于某一项参数了，因为有可能在下一次训练时这个参数就被忽略掉了。所以神经网络必须要总结出一个额外的表达式以确保部分信息被学习和保留下来。一旦某个激活值被忽略了，总有其他的激活值可以起到类似的作用。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/18%20-%20Dropout.mp4_000058.413.jpg)

看起来这种学习的方法非常低效，但它确实是一个使得神经网络鲁棒性增强并有效解决overfittting的问题。如果Dropout都无法满足你的需求，那你确实是需要一个更大更复杂的神经网络了。

# Dropout的补充
当我们真正应用神经网络的时候，我们当然不喜欢Dropout这种随机性，我们喜欢确定性更强的东西。我们通过激活值的平均值来得到一个综合的评估。下图中的Ye是所有训练时得到的Yt的平均值，我们希望在evaluation中Ye与Yt的平均值相同，这样能够保证网络的性能或行为是相同的。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/19%20-%20Dropout%20Pt.%202.mp4_000003.200.jpg)

一个小技巧就是将没有被忽略的激活值全部乘以2，在evaluation中只需要删除丢弃的值并合理缩放其他值即可得到一个被合理缩放的激活值的均值。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-03_Regularization_and_Dropout/19%20-%20Dropout%20Pt.%202.mp4_000037.092.jpg)


# TensorFlow中的Dropout
TensorFlow提供了tf.nn.dropout()函数来供我们在神经网络中集成Dropout方法

```
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```
tf.nn.dropout()函数需要两个参数：
1. hidden_layer，想要应用Dropout的Tensor
2. keep_prob，保留激活值的单元

训练时使用keep_prob=0.5，使Dropout方法丢弃一半参数


```
sess.run(logits, feed_dict={keep_prob:0.5}
```
不过要记得在测试的时候把它置为0


# 完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/blob/master/Class1/4-03_Regularization_and_Dropout/20%20-%20solutiong.py)

# 更多文章请关注我的个人网站：
[weiweizhao.com](http://weiweizhao.com/category/ai/)