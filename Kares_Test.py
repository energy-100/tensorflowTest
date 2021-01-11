import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性回归
def linearRegression():
    data=pd.read_csv('Data/Income1.csv')
    x=data.Education
    y=data.Income
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
    model.add(tf.keras.layers.Dense(3,input_shape=(2,)))
    print(model.summary())
    model.compile(optimizer='adam',loss='MSE')
    history = model.fit(x,y,epochs=5000)
    print(history)
    print(model.predict(x))

# 多层感知机
def MultilayerPercetron():
    data=pd.read_csv('Advertising.csv')
    x=data.iloc[:,1:-1]
    y=data.iloc[:,-1]
    model=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'),tf.keras.layers.Dense(1)])
    print(model.summary())
    model.compile(optimizer='adam',loss='MSE')
    history=model.fit(x,y,epochs=5000)
    print(history)

# 逻辑回归
def logisticRegression():
    # 信用卡欺诈数据
    data=pd.read_csv('Data/credit-a.csv', header=None)    #因为数据第一行没有数据，不加header参数会自动将第一行数据作为标题
    print(data.head)
    print(data.iloc[:,-1].value_counts()) #输出最后一行值的数量,记得使用iloc方法取数据
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1].replace(-1,0)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))
    print(model.summary())
    # 二分类对数损失 'binary_crossentropy'
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])#metrics=['acc']参数是指定每一步输出的信息，是列表类型
    history=model.fit(x,y,epochs=100)
    print(history)
    plt.plot(history.history['acc'])
    plt.show()

# softmax多分类问题
def softmax_MultiCategoryClassification():
    (train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()
    # plt.imshow(train_image[0])
    # plt.show()
    train_image=train_image/255
    test_image=test_image/255
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])#metrics=['acc']参数是指定每一步输出的信息，是列表类型
    history=model.fit(train_image,train_label,epochs=5)
    print(history)
    #两种损失函数 sparse_categorical_crossentropy
    pass

# softmax多分类问题
# 1.使用dropout防止过拟合
# 2.添加测试数据集
# 3.使用小型网络网址
def softmax_dropout_MultiCategoryClassification():
    (train_image,train_label),(test_image,test_label)=tf.keras.datasets.fashion_mnist.load_data()
    # plt.imshow(train_image[0])
    # plt.show()
    train_image=train_image/255
    test_image=test_image/255
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    # model.add(tf.keras.layers.Dense(128,activation='relu'))
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Dense(128,activation='relu'))
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Dense(128,activation='relu'))
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Dense(10,activation='softmax'))
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),loss='sparse_categorical_crossentropy',metrics=['acc'])#metrics=['acc']参数是指定每一步输出的信息，是列表类型
    history=model.fit(train_image,train_label,epochs=20,validation_data=(test_image,test_label))
    print(history)
    plt.plot(history.epoch,history.history['loss'],label="loss")#训练集损失
    plt.plot(history.epoch,history.history['val_loss'],label="val_loss")#测试集损失
    plt.legend()
    plt.show()

    #两种损失函数 sparse_categorical_crossentropy
    pass

if __name__ == "__main__":
    # linearRegression()
    # MultilayerPercetron()
    # logisticRegression()
    # softmax_MultiCategoryClassification()
    softmax_dropout_MultiCategoryClassification()