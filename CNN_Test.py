import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
import os


# 卷积神经网络 测试在 kaggle 上运行
# 1.卷积->池化->卷积->全局池化->输出
# 卷积核数量最好是2^n
# 最后的池化函数用 GlobalAveragePooling2D() 将每个二维参数转化为一个数据点
def CNN_demo1():
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    train_image = np.expand_dims(train_image, -1)  # 扩充一个维度
    test_image = np.expand_dims(test_image, -1)  # 扩充一个维度
    train_ds=tf.data.Dataset.from_tensor_slices((train_image, train_label))
    test_ds=tf.data.Dataset.from_tensor_slices((test_image, test_label))
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=train_image.shape[1:],activation='relu'))
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=train_image.shape[1:],activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10,  activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    history=model.fit(train_image,train_label,epochs=10,validation_data=(test_image, test_label))
    print(history)


# 卫星图片二分类
# 数据预处理部分
# 1. 数据的批量读取
# 2. 统一数据大小
# 3. 数据类型转换
# 4. 数据归一化
# 5. 样本+标签整合
# 6. 划分训练集、测试集
# 构建模型
# # 卷积层（卷积-BN层-激活函数）->池化层->...->全局均值池化层->Dense层
def CNN_satelliteImage_Classification():
    # 读取图片函数
    def load_image(path):
        img_raw = tf.io.read_file(path)  # 二进制读取
        img_tensor = tf.image.decode_jpeg(img_raw,channels=3)  # 解码二进制 channels 根据图片类型设定channels大小
        img_tensor=tf.image.resize(img_tensor,[256,256]) #统一图片尺寸
        img_tensor = tf.cast(img_tensor, tf.float32)  # 数据类型转换
        # img_tensor.numpy() #转换成numpy
        img_tensor = img_tensor / 255   #归一化
        return img_tensor


    all_image_path=glob.glob('Data/2_class/*/*.jpg')
    print(all_image_path)
    random.shuffle(all_image_path) #乱序
    label_to_index={'airplane':0,'lake':1}
    index_to_label=dict((v,k) for k,v in label_to_index.items())
    all_labels=[label_to_index[elem.split('\\')[1]] for elem in all_image_path]
    img_ds=tf.data.Dataset.from_tensor_slices(all_image_path) #样本数据
    img_ds = img_ds.map(load_image)  #作用map()函数，将路径转化为图像数据
    label_ds=tf.data.Dataset.from_tensor_slices(all_labels) #标签数据
    img_label_ds=tf.data.Dataset.zip((img_ds,label_ds)) #样本数据+标签数据

    train_count=int(len(all_image_path)*0.2)
    test_count=len(all_image_path)-train_count
    train_ds=img_label_ds.skip(test_count)
    test_ds=img_label_ds.take(test_count)

    BATCH_SIZE=16
    train_ds=train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
    test_ds=test_ds.batch(BATCH_SIZE)

    model=keras.Sequential()
    model.add(keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(128,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(128,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(256,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(256,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512,(3,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.GlobalAveragePooling2D()) #对每层（通道）图像求平均 得到通道数量的数据数量
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    print(model.summary())

    # # 多分类使用CategoricalCrossentropy()函数，且模型最后没有使用逻辑激活函数(sigmoid或softmax函数)激活时 from_logits=True
    # # 标签为非 noe-hot编码时 使用 SparseCategoricalCrossentropy()函数
    # model.compile(optimizer=keras.optimizers.Adam(0.01),loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    model.compile(optimizer=keras.optimizers.Adam(0.01),loss=keras.losses.Binary_crossentropy(),metrics=['acc']) # loss函数名称 要大写
    history=model.fit(train_ds,epoch=10,validation_data=test_ds,steps_per_epoch=train_count//BATCH_SIZE,validation_steps=test_count//BATCH_SIZE)
    print(history)

    #预测
    img_path='test01.jpg'
    img=load_image(img_path)
    img=tf.expand_dims(img,axis=0) # 扩展维度，由于训练时模型输入为4D batch-高-宽-通道，因此预测数据应该也为4D
    pred=model.predict(img)
    print(index_to_label[(pred>0.5).astype('int')[0][0]])  #predict返回结果是二维结构，样本数-每个输出节点数值

    # # 随机选取一张图片显示
    # i=random.choice(range(len(all_image_path)))
    # image_path=all_image_path[i]
    # tensor_image=load_image(image_path)
    # plt.title(index_to_label[label1[i]])
    # plt.imshow(tensor_image.numpy())
    # plt.show()

# 学习模型最后直接输出（不使用sigmoid或softmax激活） from_logits=True 这样计算更加稳定
def CNN_birds_Classification():
    # 读取图片函数
    def load_image(path,label):
        img_raw = tf.io.read_file(path)  # 二进制读取
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)  # 解码二进制 channels 根据图片类型设定channels大小
        img_tensor = tf.image.resize(img_tensor, [256, 256])  # 统一图片尺寸
        img_tensor = tf.cast(img_tensor, tf.float32)  # 数据类型转换
        # img_tensor.numpy() #转换成numpy
        img_tensor = img_tensor / 255  # 归一化
        return img_tensor,label

    def load_preprocess_image(path,label):
        img_raw = tf.io.read_file(path)  # 二进制读取
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)  # 解码二进制 channels 根据图片类型设定channels大小
        img_tensor = tf.image.resize(img_tensor, [256, 256])  # 统一图片尺寸
        img_tensor = tf.cast(img_tensor, tf.float32)  # 数据类型转换
        # img_tensor.numpy() #转换成numpy
        img_tensor = img_tensor / 255  # 归一化
        return img_tensor,label

    all_image_path = glob.glob('Data/birds/*/*.jpg')
    # print(all_image_path)

    all_label_names = [v.split('\\')[1] for v in all_image_path]    #所有标签
    label_names = set(all_label_names)  #去重后的所有标签
    label_to_index = dict((elem,i) for i,elem in enumerate(label_names))
    index_to_label = dict((v,k) for k,v in label_to_index.items())
    # print([label_to_index[label] for label in all_label_names[-5:]])
    np.random.seed(10)
    random_index = np.random.permutation(len(all_label_names)) #讲数据乱序
    img_path=np.array(all_image_path)[random_index]
    label_path=np.array(all_label_names)[random_index]
    i= int(len(all_image_path)*0.8)
    train_path=img_path[:i]
    train_label=label_path[:i]
    test_path=img_path[i:]
    test_label=label_path[i:]
    train_ds=tf.data.Dataset.from_tensor_slices((train_path,train_label))
    test_ds=tf.data.Dataset.from_tensor_slices((test_path,test_label))

    train_ds=train_ds.map(load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)  #自动选择读取图片时使用线程数
    test_ds=test_ds.map(load_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)  #自动选择读取图片时使用线程数

    BATCH_SIZE = 32
    train_ds = train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)


    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(128, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(128, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(256, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(256, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(512, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(512, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(512, (3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.GlobalAveragePooling2D())  # 对每层（通道）图像求平均 得到通道数量的数据数量
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    # # 多分类使用CategoricalCrossentropy()函数，且模型最后没有使用逻辑激活函数(sigmoid或softmax函数)激活时 from_logits=True，数值的变换处理会在损失函数这里处理
    # # 标签为非 noe-hot编码时 使用 SparseCategoricalCrossentropy()函数
    # # 再多分类问题中，一般将在最后直接输出（不使用sigmoid或softmax激活） from_logits=True 这样计算更加稳定
    # model.compile(optimizer=keras.optimizers.Adam(0.01),loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])  # loss函数名称 要大写
    history = model.fit(train_ds, epoch=10, validation_data=test_ds, steps_per_epoch=len(train_path) // BATCH_SIZE,
                        validation_steps=len(test_path)  // BATCH_SIZE)
    print(history)

    # 预测
    img_path = 'test01.jpg'
    img = load_preprocess_image(img_path)
    img = tf.expand_dims(img, axis=0)  # 扩展维度，由于训练时模型输入为4D batch-高-宽-通道，因此预测数据应该也为4D
    pred = model.predict(img)
    print(index_to_label[np.argmax(pred)])  # predict返回结果是二维结构，样本数-每个输出节点数值

    # # 随机选取一张图片显示
    # i=random.choice(range(len(all_image_path)))
    # image_path=all_image_path[i]
    # tensor_image=load_image(image_path)
    # plt.title(index_to_label[label1[i]])
    # plt.imshow(tensor_image.numpy())
    # plt.show()


if __name__ == "__main__":
    # CNN_demo1()
    # CNN_satelliteImage_Classification()
    CNN_birds_Classification()