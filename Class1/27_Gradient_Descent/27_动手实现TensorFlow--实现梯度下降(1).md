上节实现了误差计算之后我们已经实现所有前向的计算步骤。下面我们将使用计算出的误差进行反向的操作，也就是反向传播。在这个过程中，神经网络计算权重该如何修改才能够使得误差最小。我们通常使用梯度下降法实现对权重的修改。
<!--more-->
# 梯度下降

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/01%E4%B8%89%E7%BB%B4%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D.png) 

想象在三维空间的表面放置一个小球，如上图。小球的高度代表神经网络输出与实际的误差，每一个维度表示输入变量，有m个输入代表m个维度。在最理想状态下，小球应该处于谷底，代表神经网络的输出与实际值之间的误差最小。
在初始状态下，小球被随机放置。梯度下降的工作原理是先计算当前点平面的斜率,其中包括计算所有参数损失的部分导数。这组部分导数称为梯度。然后使用梯度来修改权重,以便下一个向前传递的输出误差更低。也就是在球的位置测量山谷的坡度,然后在坡度方向上小幅度移动小球。随着时间的推移, 可以通过很多小的移动来找到谷底。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/02%E5%B0%8F%E5%B9%85%E7%A7%BB%E5%8A%A8%E5%B0%8F%E7%90%83.png)

虽然梯度下降原理上可以找到一个最低点，但是这个技术却不能找到一个**绝对的**最低点。这是因为它可能会被困在局部最低点, 而绝对最低点可能是  "在下一个山丘 "。

# 在MiniFlow中集成梯度下降
我们已经知道我们需要将小球推到谷底，但是回到我们的代码，我们如何通过Cost函数找到谷底的方向？其实梯度本身就代表了实际的信息。定义上梯度是代表最陡的上坡方向（微分，斜率），如果我们在它前面加一个负号“-”，我们就可以得到最陡的下坡方向。

下一个问题是我们每次推动小球应该推多少，这就是Learning rate，代表神经网络的学习速度。然而这个速度并不是越快越好，如果这个值过大会使得神经网络会错过最小误差

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/03gradient-descent-convergence.gif)

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/04gradient-descent-divergence.gif)

那么好的learning rate应该是什么样的？
通常learning rate被设置在0.1~0.0001之间，其中0.001~0.0001最常用。

梯度下降公式的伪代码：
```
x = x - learning_rate * gradient_of_x
#x表示神经网络中的某个参数，例如权重weight或者偏置bias
```

# 梯度下降的代码实现

```
def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.
    x = x - learning_rate*gradx
    # Return the new value for x
    return x

```

测试代码：

```
"""
Given the starting point of any `x` gradient descent
should be able to find the minimum value of x for the
cost function `f` defined below.
"""
import random
from gd import gradient_descent_update


def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5


def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x


# Random number better 0 and 10,000. Feel free to set x whatever you like.
x = random.randint(0, 10000)
# TODO: Set the learning rate
learning_rate = 0.1
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)

```

测试结果(*纵坐标是x的值的更新过程*)：
learning rate=0.1

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.png)

learning rate=1

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/27_Gradient_Descent/learning_rate_1.png)

完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/tree/master/Class1/27_Gradient_Descent)

更多文章请关注我的个人网站：
[link](http://weiweizhao.com/category/ai/)


