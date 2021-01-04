import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性回归
def linearRegression():
    data=pd.read_csv('income1.csv')
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
    data=pd.read_csv('credit-a.csv',header=None)    #因为数据第一行没有数据，不加header参数会自动将第一行数据作为标题
    print(data.head)
    print(data.iloc[:,-1].value_counts()) #输出最后一行值的数量,记得使用iloc方法取数据
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1].replace(-1,0)
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])#metrics=['acc']参数是指定每一步输出的信息，是列表类型
    history=model.fit(x,y,epochs=100)
    print(history)
    plt.plot(history.history['acc'])
    plt.show()

# softmax多分类问题
def softmax_MultiCategoryClassification():
    pass

if __name__ == "__main__":
    # linearRegression()
    # MultilayerPercetron()
    logisticRegression()