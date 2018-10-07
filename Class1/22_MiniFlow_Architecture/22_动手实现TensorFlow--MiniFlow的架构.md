本文通过Python代码实现一个仿TensorFlow的神经网络框架——MiniFlow，该框架具备TensorFlow的核心特征，反向传播和计算图。
在实现MiniFlow之前我们先看一下计算图的含义。
# 什么是神经网络
神经网络是数学函数的图，例如由线性组合和激活函数，该图由节点和边组成。

![image](http://note.youdao.com/favicon.ico)

每一层的节点（输入层节点除外）使用一个数学函数表示，将上一层的输出进行运算。例如一个节点可以用f(x,y)=x+y表示，x和y分别是上一层节点的输出。

同样的，每一个节点设置一个输出值，可以传递到下一层。在输入层和输出层之间的层都称为隐藏层。（Hidden Layer）

## 前向传播Forward Propagation
把输入值从输入层通过整个网络传播，穿过所有节点的数学函数，最终输出一个值，这个过程就叫前向传播

## 图
节点和边沿创建了一个图结构。通过前面的章节我们可以理解，图的结构越复杂，可以解决的问题越复杂。

创建神经网络一般有两个步骤：
1. 定义一个图，含节点（nodes）和边沿（edges）
2. 将数值通过图进行传播

我们的MiniFlow也是同样的实现方式。

# MiniFlow的架构
我们使用一个Python 类（class）来表示节点（nodes）

```
class Node(object):
    def __init__(self):
        # Properties will go here!
```
我们知道每个节点可能从其他多个节点接收数据，同时每个节点只有一个输出，传递到其他节点。所以我们需要这个节点有两个列表：
1. inbound 用于存储输入关系
2. outbound用于存储输出关系

```
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
```
每个节点最终都需要计算它的输出值，但初始化的时候首先将创建一个空的参数，初始化为None。

```
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None
```
每一个节点需要能给进行前向和反向传播，现在我们先定义一个前向传播的函数，后续我们再处理反向传播

```
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented
```
## 节点的计算
Node类定了一些基础的每一个节点都具备的特征。图结束的节点（输出层节点）有一些特殊的特征需要用子类来描述。下面我们实现一个Node类的输入子类

```
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value
```
与其他Node的子类不同，这个input子类事实上没有做任何计算，这个子类仅仅用于保存数据。你可以通过forward()方法显式设置这个值。这个值可以输入到神经网络的其余节点。

## 加法子类（ADD）
add类是Node类的另一个子类，可以实现累加。

```
class Add(Node):
    def __init__(self, x, y):
        # You could access `x` and `y` in forward with
        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
        Node.__init__(self, [x, y])

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of it's inbound_nodes.
        
        Your code here!
        """
        self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value
```
# 总结
本文实现了神经网络MiniFlow的基础架构：
1. 通用节点类node class,包含输入、输出、值、前向传播函数
2. 输入节点类input class，为node的子类，只保存数据，不做计算
3. 加法节点类add class，为node的子类，做累加计算



