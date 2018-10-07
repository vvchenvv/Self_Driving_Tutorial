在MiniFlow中我们设置了两种方法使数据在整个图中传递：topological_sort()和forward_pass()。topological_sort()是对图中的节点计算顺序进行排序，使节点按照某种有规律的顺序进行计算。

# 拓扑排序topological_sort()
为了定义一个网络，我们必须对节点计算顺序进行排序。鉴于某个节点的输入取决于其他节点的输出，我们需要以这样的方式展平图形，在尝试运行某个节点的计算之前解析每个节点的所有输入依赖性。这是一种称为=拓扑排序==的技术。

[**拓扑排序wiki**](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm)

![image](http://note.youdao.com/favicon.ico)

MiniFlow中的拓扑排序算法是Kahn算法。

## Kahn算法
以下是Kahn算法的伪代码

```
L ← Empty list that will contain the sorted elements
S ← Set of all nodes with no incoming edge
while S is non-empty do
    remove a node n from S
    add n to tail of L
    for each node m with an edge e from n to m do
        remove edge e from the graph
        if m has no other incoming edges then
            insert m into S
if graph has edges then
    return error   (graph has at least one cycle)
else 
    return L   (a topologically sorted order)
```
不难看出该算法的实现十分直观，关键在于需要维护一个入度为0的顶点的集合：

每次从该集合中取出(没有特殊的取出规则，随机取出也行，使用队列/栈也行，下同)一个顶点，将该顶点放入保存结果的List中。

紧接着循环遍历由该顶点引出的所有边，从图中移除这条边，同时获取该边的另外一个顶点，如果该顶点的入度在减去本条边之后为0，那么也将这个顶点放到入度为0的集合中。然后继续从集合中取出一个顶点…………

当集合为空之后，检查图中是否还存在任何边，如果存在的话，说明图中至少存在一条环路。不存在的话则返回结果List，此List中的顺序就是对图进行拓扑排序的结果。

感兴趣的伙伴们可以参考博客[拓扑排序的两种实现：Kahn算法和dfs算法](https://blog.csdn.net/qinzhaokun/article/details/48541117)

![image](http://note.youdao.com/favicon.ico)

对上图进行拓扑排序的结果：
2->8->0->3->7->1->5->6->9->4->11->10->12
# 前向传播forward_pass()
这是实际运行整个网络的函数，最终输出结果。这个函数需要根据topological_sort()函数输出的已排序的节点序列计算每个节点。

```
def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
```

# 实际运行
miniflow.py文件可参考github链接[代码链接](http://note.youdao.com/)
nn.py创建一个有两个输入节点，一个add节点的网络并运行。网络结构如图：

![image](http://note.youdao.com/favicon.ico)


```
from miniflow import *

x, y = Input(), Input()

# f = Add(Add(Add(x, y),y),x)
f=Add(x,y)

feed_dict = {x: 10, y: 5}

sorted_nodes = topological_sort(feed_dict)
# print(len(sorted_nodes))
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
# print("(({} + {}) + {}) + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y],feed_dict[y], feed_dict[x], output))
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))

```

实际运行结果：
> 10 + 5 = 15 (according to miniflow)

小伙伴们可以把注释掉的两行代码取消注释，同时注释掉他们相同功能的代码，看看会发生什么。


```
# f = Add(Add(Add(x, y),y),x)
# print("(({} + {}) + {}) + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y],feed_dict[y], feed_dict[x], output))
```

这里实际上会创建一个三个add节点加两个输入节点的图，具体过程请小伙伴们根据代码推断。
