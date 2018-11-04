本章节开始学习如何使用TensorFlow解决实际问题。围绕MNIST数据集识别出图像中的字母作为目标。以下是MNIST数据集中字母A的一些图形：

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-00_Install_Tensorflow/01MNIST%E6%95%B0%E6%8D%AE%E9%9B%86.jpg)

我们的目标是使神经网络自动识别出图像代表的是什么字母。首先我们在电脑上安装TensorFlow

# 安装TensorFlow
## OS X or Linux
### 系统需求
- Python 3.4 or higher 
- Anaconda

###  Install TensorFlow

```
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow
```

## Windows
### Install Docker
[点击链接安装Docker https://docs.docker.com/engine/installation/windows/](https://docs.docker.com/engine/installation/windows/)

### 拉取TensorFlow的docker镜像
由于墙的原因，我们无法在线使用存放在Google cloud上面的镜像，需要首先下载到本地：

```
docker pull tensorflow/tensorflow
```
等待下载完成

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-00_Install_Tensorflow/02Pull-Docker.PNG)

### Run the Docker Container
通过以下命令启动一个带TensorFlow的jupyter notebook

```
docker run -it -p 8888:8888 tensorflow/tensorflow
```
这个docker镜像里面包含了三个TensorFlow例子的notebook，我们也可以建立自己的notebook。通过Localhost：8888可以访问这个notebook。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-00_Install_Tensorflow/03%E8%BF%90%E8%A1%8Cdocker.PNG)

# Hello World
在jupyter中新建notebook，运行以下代码测试安装是否成功

```
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

运行结果：

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/3-00_Install_Tensorflow/05Helloworld.PNG)

完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/tree/master/Class1/3-00_Install_Tensorflow)

更多文章请关注我的个人网站：
[link](http://weiweizhao.com/category/ai/)