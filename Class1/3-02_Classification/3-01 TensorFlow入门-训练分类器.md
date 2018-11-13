分类事一种给定输入和标记的任务，如下图的字母，每个字母有一个标签，说明它是什么。典型情况下我们会有很多样本，我们把它归类后交给分类器学习。当出现了全新的样本时，分类器的目标是指出这个新样本属于哪一类。虽然机器学习不仅仅包含分类，但是分类是我们机器学习的基础，例如：排序，目标检测，回归等算法都需要基于分类。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/10.%20Supervised%20Classification.mp4_000017.582.jpg)

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/10.%20Supervised%20Classification.mp4_000024.613.jpg)

# 训练逻辑分类器
逻辑分类器是一种线性分类器，它接收输入（例如某个图像的像素），对输入执行一个线性函数来生成预测

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/12.%20Training%20Your%20Logistic%20Classifier.mp4_000017.658.jpg)

线性函数实际上是一个大的矩阵相乘，它把所有输入当成一个大的矢量X来表示乘以每个输入对应的权重W矩阵来生成预测矩阵。权重和偏置都是机器学习所要调整的参数。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/12.%20Training%20Your%20Logistic%20Classifier.mp4_000041.933.jpg)

训练的过程就是不断调整权重和偏置，使得预测的结果误差最小。以上线性函数的输出值怎么来执行分类呢？

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/12.%20Training%20Your%20Logistic%20Classifier.mp4_000051.381.jpg)

我们把每张图片当做输入且只有一个标签，因此我们将这些结果转换为概率：

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/12.%20Training%20Your%20Logistic%20Classifier.mp4_000104.126.jpg)

使得正确的分类的概率非常接近1，其他分类概率接近于0.把结果转换为概率的方法是Softmax方法
## Softmax方法
关于Softmax函数的介绍可以参考百度百科：
[Softmax函数](https://baike.baidu.com/item/Softmax%E5%87%BD%E6%95%B0/22772270?fr=aladdin)

这个函数的特点是它的输出结果之和一定为1。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/12.%20Training%20Your%20Logistic%20Classifier.mp4_000000.000.jpg)

# TensorFlow的线性方程
线性方程y = Wx + b，我们需要将我们的输入X转换为我们的标签Y。以图像分类为例，我们想把图像对应到数字上，识别出图像上的数字。

X是我们的像素点列表，y是归类，每一个归类代表一个数字。首先看看y=wX，也就是权重如何影响x对y的预测。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/y%3Dwx%E5%87%BD%E6%95%B0.jpg)

y=wX已经可以让我们把数据分类到不同的标签内了，然而这个公式还有个问题就是当x输入为0的时候y一定输出0，这与实际不符。我们希望能够避开这个问题，因此我们引入了偏置b

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/y%3Dwx%2Bb%E5%87%BD%E6%95%B0.jpg)

y=wX+b允许我们去创建预测了。

## 矩阵的转置
我们之前说我们的函数是y=wX+b，然而另一个函数也能实现相应的功能：y=Xw+b。需要注意的是：由于我们的w、X、b都是矩阵形式，这两个顺序其实是不一样的。
y=wX+b

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/y%3Dwx%2Bb%E5%87%BD%E6%95%B0-%E7%A4%BA%E4%BE%8B.jpg)

y=Xw+b

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/y%3DXw%2Bb%E5%87%BD%E6%95%B0-%E7%A4%BA%E4%BE%8B.jpg)

可以看得出来上面两个公式的矩阵都进行了转置。x现在的维度是一个1行3列（1x3），w的维度是3行2列（3x2），b的维度是1行2列（1x2）。计算结果会输出一个1行2列的矩阵。我们可以注意到这个1行2列的结果和上一个公式2行1列的结果值是一样的，只是进行了转置。这两个值就代表了我们的分类。

# TensorFlow中的权重和偏置
我们之前说过神经网络的目标是修改权重和偏置，那之前我们所学的 tf.placeholder()和tf.constant()就不适用了，因为这两个的值都无法修改，我们需要用到 tf.Variable 。

```
x = tf.Variable(5)
```
tf.Variable创建一个带有初始值的Tensor，就和我们Python里的变量很像。所有的变量都必须进行初始化，我们可以用tf.global_variables_initializer()对所有变量进行初始化。

```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

所有的权重和偏置变量都会被随机初始化，避免了每次训练时我们的模型都从同一个地方开始的尴尬。同样的，从正态分布中选择权重可防止任何一个权重压倒其他权重。我们将使用tf.truncated_normal() 函数来生成符合正态分布的随机数。


```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```
tf.truncated_normal函数返回一个Tensor，这个Tensor具有正太分布的随机值，它的幅度和平均值的偏差不超过2个标准差。由于权重已经被随机初始化了，我们不用担心每次训练都从同一个地方开始，因此偏置值不需要随机初始化，可以全部设置初始值为0.


```
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

# 实战：识别手写0,1,2
我们从MNIST数据集训练入手，识别出手写的数字0,1,2

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-02_Classification/012%E9%83%A8%E5%88%86%E6%89%8B%E5%86%99%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)

## 代码

```
# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    return weights
    pass


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    bias = tf.Variable(tf.zeros(n_labels))
    return bias
    pass


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    logits = tf.add(tf.matmul(input, w), b)
    return logits
    pass
```

这里需要注意的是矩阵乘法使用的是tf.matmul，而且顺序很重要

# 完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/tree/master/Class1/3-02_Classification)

# 更多文章请关注我的个人网站：
[link](http://weiweizhao.com/category/ai/)



