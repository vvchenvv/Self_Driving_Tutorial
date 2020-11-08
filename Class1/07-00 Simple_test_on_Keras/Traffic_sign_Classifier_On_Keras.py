# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
#from generator import generator

# WAITING FOR CODE PACKAGE TO SYNC UP
with open('train.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical

# # Build Convolutional Pooling Neural Network with Dropout in Keras Here
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(43))
# model.add(Activation('softmax'))

#LeNet
def LeNet(X_train,Y_train):
    model=Sequential()
    model.add(Conv2D(filters=5,kernel_size=(3,3),strides=(1,1),input_shape=X_train.shape[1:],padding='same',
                     data_format='channels_last',activation='relu',kernel_initializer='uniform'))  #[None,28,28,5]
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2,2)))  #池化核大小[None,14,14,5]

    model.add(Conv2D(16,(3,3),strides=(1,1),data_format='channels_last',padding='same',activation='relu',kernel_initializer='uniform'))#[None,12,12,16]
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(2,2))  #output_shape=[None,6,6,16]

    model.add(Conv2D(32, (3, 3), strides=(1, 1), data_format='channels_last', padding='same', activation='relu',
                     kernel_initializer='uniform'))   #[None,4,4,32]
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(100,(3,3),strides=(1,1),data_format='channels_last',activation='relu',kernel_initializer='uniform'))  #[None,2,2,100]
    model.add(Flatten(data_format='channels_last'))  #[None,400]
    model.add(Dense(168,activation='relu'))   #[None,168]
    model.add(Dense(84,activation='relu'))    #[None,84]
    model.add(Dense(43,activation='softmax'))  #[None,10]
    #打印参数
    model.summary()
    #编译模型
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


                                                                                     
# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model = LeNet(X_normalized,y_one_hot)
model.fit(X_normalized, y_one_hot, epochs=10, validation_split=0.2)
# model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# history = model.fit(X_normalized, y_one_hot, epochs=10, validation_split=0.2)

with open('test.p', 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))