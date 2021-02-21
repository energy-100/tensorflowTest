import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def ONED():
    data = pd.read_csv('Data/leafData/train.csv')
    labels=pd.factorize(data.species)[0] # factorize返回两个数组 一个存储特征所有可能值的序号，一个存储所有可能值
    x = data[data.columns[2:]] # 前两列丢弃
    train_x,test_x,train_y,test_y=train_test_split(x,labels)
    print(train_x)
    mean=train_x.mean(axis=0)
    std=train_x.std(axis=0)
    train_x=(train_x-mean)/std
    test_x=(test_x-mean)/std
    train_x=np.expand_dims(train_x,-1)
    test_x=np.expand_dims(test_x,-1)

    # 一维卷积输入格式 samples feature 每个特征的通道数
    model=keras.Sequential()
    model.add(keras.layers.Conv1D(32,7,input_shape=(train_x.shape[1:]),activation='relu',padding='same'))  # 32 为卷积核个数 7为卷积核长度
    model.add(keras.layers.Conv1D(32,7,activation='relu',padding='same'))
    model.add(keras.layers.MaxPooling1D(3))  # 讲妹每个feature map 中的三个特征合成一个
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(64,7,activation='relu',padding='same'))
    model.add(keras.layers.Conv1D(64, 7, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(3))  # 讲妹每个feature map 中的三个特征合成一个
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(128,7,activation='relu',padding='same'))
    model.add(keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(3))  # 讲妹每个feature map 中的三个特征合成一个
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(256,7,activation='relu',padding='same'))
    model.add(keras.layers.Conv1D(256, 7, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(3))  # 讲妹每个feature map 中的三个特征合成一个
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256))  # 讲妹每个feature map 中的三个特征合成一个
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(99,activation='softmax'))
    print(model.summary())
    # 可进一步增加网络深度 并使用残差网络结构避免梯度消失



    model.compile(optimizer=keras.optimizers.RMSprop(),loss='sparse_categorical_crossentropy',metrics=['acc']) #非one-hot编码使用sparse
    # history = model.fit(train_x, train_y, epochs=600, batch_size=128, validation_data=(test_x, test_y))
    history = model.fit(train_x, train_y, epochs=1000, validation_data=(test_x, test_y)) # 每个epoch训练所有数据
    plt.plot(history.epoch,history.history['acc'],'y',label='training')
    plt.plot(history.epoch,history.history['val_acc'],'b',label='test')
    plt.legend()
    plt.show()
    # print(history)




if __name__ == "__main__":
    ONED()