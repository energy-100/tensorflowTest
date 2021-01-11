import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers


# def key_hot_test():
#     data = pd.read_csv('Data/Tweets.csv').text
#     for
#     print(data)
#     # for i in range(data.shape[0]):
#     #     for j in
#     #     data[i][1:]


# 处理序列化数据
# 1.将样本长度统一
# 2.通过Embedding层进行降维
# 3.使用Fatten()层或GlobalAveragePooling1D()层扁平化
# 4.增加全连接层
# 4.使用dropout层防止过拟合
# 5.编译，训练
def CNN_imdb_Test():
    data=keras.datasets.imdb
    MAX_WORD=1000
    (x_train,y_train),(x_test,y_test)=data.load_data(num_words=MAX_WORD)
    x_train=keras.preprocessing.sequence.pad_sequences(x_train,300)  #将每条数据限制的长度全部变为300
    x_test=keras.preprocessing.sequence.pad_sequences(x_test,300)
    model=keras.Sequential()
    model.add(layers.Embedding(10000,50,input_length=300))  # 输入的词汇表的大小 输出维度（降维后的维度）
    # model.add(layers.Flatten())  # 输入的词汇表的大小 输出维度（降维后的维度）
    model.add(layers.GlobalAveragePooling1D())  # 输入的词汇表的大小 输出维度（降维后的维度）
    model.add(layers.Dense(128,activation='relu'))  # 输入的词汇表的大小 输出维度（降维后的维度）
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1,activation='sigmoid'))  # 输入的词汇表的大小 输出维度（降维后的维度）
    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(0.001),loss=keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    history=model.fit(x_train,y_train,epochs=15,batch_size=256,validation_data=(x_test,y_test))
    print(history)
if __name__ == "__main__":
    # key_hot_test()
    CNN_imdb_Test()