我们在上节课的训练中发现，每个回合训练的时间很久，加大训练回合之后所用的训练时间就更长了。一旦我们的程序结束，再次运行时原有的weight和bias信息全部消失了，又要重新训练。TensorFlow中设置了保存与加载的机制来解决这个问题。同时我们上节课建立的神经网络还不够“深”，只有一个隐藏层，这节课我们来加深神经网络。

# 加深神经网络
上节课我们一开始使用epoch=20，learning rate=0.001，训练结束后的精度accuracy=0.839

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-02_Save_and_restore_tensorflow_models/2%E5%B1%82%E7%BD%91%E7%BB%9C%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.png)

我们只用了一层ReLU，如果我们加大神经网络的复杂度，再增加一层ReLU，那这个精度表现会不会有改变呢？试试就知道了


```
# 权重和偏置需要有两份，一份是wx+b；另一份是w1*ReLU输出+b1
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
```

这里面我们增加了一层hidden_layer2，系统示意图如下：

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-02_Save_and_restore_tensorflow_models/%E4%B8%A4%E5%B1%82ReLU.PNG)

这样我们就建立了3层的神经网络。运行一次训练会发现accuracy直接提升到了0.92以上，证明增加网络层数对提升精度，解决复杂问题是非常有利的。

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-02_Save_and_restore_tensorflow_models/3%E5%B1%82%E7%BD%91%E7%BB%9C%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.png)


我们可以看到在3层神经网络训练过程中cost的波动远大于2层神经网络，但是最终的accuracy却高了很多，这是一个有趣的现象，大家可以自由调节参数看看是什么原因导致的。

# 保存神经网络
TensorFlow中保存的机制是tf.train.Saver.save() 方法，这个方法会将模型中的weight和bias全部保存下来，并且可以保存模型结构。

```
save_file = './train_model.ckpt'
saver = tf.train.Saver()

<----------训练过程------------->

saver.save(sess, save_file)
print('Trained Model Saved.')
```
输出结果：

```
Epoch: 0001 cost= 118.16108703
Epoch: 0002 cost= 35.628852844
Epoch: 0003 cost= 33.554569244
Epoch: 0004 cost= 25.575164795
Epoch: 0005 cost= 16.287216187
Epoch: 0006 cost= 33.322113037
Epoch: 0007 cost= 27.648113251
Epoch: 0008 cost= 41.859161377
Epoch: 0009 cost= 10.666368484
Epoch: 0010 cost= 22.679353714
Epoch: 0011 cost= 13.955465317
Epoch: 0012 cost= 27.295814514
Epoch: 0013 cost= 8.500268936
Epoch: 0014 cost= 28.089063644
Epoch: 0015 cost= 9.669015884
Epoch: 0016 cost= 8.695211411
Epoch: 0017 cost= 6.443881512
Epoch: 0018 cost= 4.647359371
Epoch: 0019 cost= 15.570621490
Epoch: 0020 cost= 8.495063782
Optimization Finished!
Accuracy: 0.9296875
Trained Model Saved.
```
在当前目录下会出现4个文件：

![image](https://raw.githubusercontent.com/vvchenvv/Self_Driving_Tutorial/master/Class1/4-02_Save_and_restore_tensorflow_models/%E4%BF%9D%E5%AD%98model-%E6%96%87%E4%BB%B6.png)

# 加载已训练的神经网络
加载的过程也比较简单，使用tf.train.Saver.restore()即可

```
        saver.restore(sess, save_file)
		# Test model
		correct_prediction = tf.equal(tf.argmax(logits, 1),tf.argmax(y, 1))
	    # Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		test_accuracy = sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print('Test Accuracy: {}'.format(test_accuracy))
		print("Model loaded!")
```
测试结果：
```
Test Accuracy: 0.9185000061988831
Model loaded!
```
这样我们就完成了模型的保存与加载，非常方便。

# 完整代码请关注我的github：
[Github链接](https://github.com/vvchenvv/Self_Driving_Tutorial/blob/master/Class1/4-02_Save_and_restore_tensorflow_models/multilayer_perceptron.py)

# 更多文章请关注我的个人网站：
[weiweizhao.com](http://weiweizhao.com/category/ai/)
