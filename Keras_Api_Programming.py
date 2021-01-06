import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#keras api编程
# 多输入多输出模型测试
# Model的官方API https://www.tensorflow.org/api_docs/python/tf/keras/Model
def Mult_input_output_demo():
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    # plt.imshow(train_image[0])
    # plt.show()
    train_image = train_image / 255
    test_image = test_image / 255
    input1 =keras.Input(shape=(28,28))
    input2 =keras.Input(shape=(28,28))
    x1=keras.layers.Flatten()(input1)
    x2=keras.layers.Flatten()(input2)
    x=keras.layers.concatenate([x1,x2])
    output=keras.layers.Dense(1,activation='sigmoid')(x)
    model=keras.Model(inputs=[input1,input2],outputs=output) #此处存疑 到底是input 还是inputs

# keras tf.data 读取数据测试
# 从list、numpy、字典、读取数据
# 使用 .take(int) 取指定元素

def keras_tf_data_loadData_demo():
    #将list或numpy数据中的每一个元素变成dataset中的一个元素（切片）

    dataset=tf.data.Dataset.from_tensor_slices([1,3,4,5,6,7,8,9,10])
    for elem in dataset:
        elemnumpy=elem.numpy() #转化成numpy类型
        print(elemnumpy)

    dataset2D = tf.data.Dataset.from_tensor_slices([[1, 3], [4, 5], [6, 7], [8, 9]]) #数据结构必须相同，不能有的有两个元素，有的有三个元素
    for elem in dataset2D.take(3):
        elemnumpy=elem.numpy() #转化成numpy类型
        print(elemnumpy)

    datasetDict = tf.data.Dataset.from_tensor_slices({'a':[1, 3],'b':[4, 5],'c':[6, 7]}) #数据结构必须相同，不能有的有两个元素，有的有三个元素
    for elem in datasetDict:
        #elemnumpy=elem.numpy() #转化成字典无法转换为numpy类型
        print(elem)

    datasetNumpy = tf.data.Dataset.from_tensor_slices(np.array([[1, 3], [4, 5], [6, 7], [8, 9]])) #数据结构必须相同，不能有的有两个元素，有的有三个元素
    for elem in datasetNumpy:
        #elemnumpy=elem.numpy() #转化成字典无法转换为numpy类型
        print(elem)

# keras tf.data 处理数据测试
# .shuuffle(buffer_size=) 打乱数据排列顺序
# .repeat(count) 重复次数
# .batch()  每次数据量大小
# .map() 对每个元素应用一个函数
def keras_tf_data_Data_Process():
    #将list或numpy数据中的每一个元素变成dataset中的一个元素（切片）
    dataset=tf.data.Dataset.from_tensor_slices([1,3,4,5,6,7,8,9,10])
    # 因为tf.data.Dataset是一个迭代器 因此无法知道其大小
    dataset.shuffle(10) # reshuffle_each_iteration默认为true 每次循环的乱序都不同
    dataset.repeat() # count参数不设置为无限循环
    dataset.batch(5)
    dataset.map(tf.square)  #对每个元素进行平方

# 读取数据实战1
def keras_tf_data_Data_Process_demo1():
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
    train_image = train_image / 255
    test_image = test_image / 255
    ds_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))
    ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label))

    # 方法2
    # ds_train_img=tf.data.Dataset.from_tensor_slices(train_image)
    # ds_train_lab=tf.data.Dataset.from_tensor_slices(train_label)
    # ds_train=tf.data.Dataset.zip((ds_train_img,ds_train_lab))

    ds_train=ds_train.shuffle(10000).repeat().batch(64)
    ds_test=ds_test.batch(64) #测试数据也要使用 batch()和训练数据一样的batch
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10,  activation='softmax'))
    print(model.summary())
    # 多分类对数损失 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])#metrics=['acc']参数是指定每一步输出的信息，是列表类型
    # history=model.fit(ds_train,epochs=5,steps_per_epoch=train_image.shape[0]//64)
    history=model.fit(ds_train,
                      validation_data=ds_test,
                      epochs=5,
                      steps_per_epoch=train_image.shape[0]//64,
                      validation_steps=test_image.shape[0]//64
                      )
    print(history)
    plt.plot(history.epoch,history.history['loss'],label="loss")#训练集损失
    plt.plot(history.epoch,history.history['val_loss'],label="val_loss")#测试集损失
    plt.plot(history.epoch, history.history['acc'], label="acc")  # 训练集损失
    plt.plot(history.epoch, history.history['val_acc'], label="val_acc")  # 测试集损失
    plt.legend()
    plt.show()
    #命令行显示的数值为batch数=数据集数量/batch_size


if __name__ == "__main__":
    # Mult_input_output_demo()
    # keras_tf_data_loadData_demo()
    # keras_tf_data_Data_Process()
    keras_tf_data_Data_Process_demo1()