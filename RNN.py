
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import random
import datetime
import pandas as pd
import re
import tensorflow.keras.layers as layers

# tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()
def RNN_demo():
    data=pd.read_csv('Data/Tweets.csv')
    data=data.loc[:,['airline_sentiment','text']]
    print(data.airline_sentiment.unique())
    print(data.airline_sentiment.value_counts())
    data_p = data[data.airline_sentiment=='positive']
    data_n = data[data.airline_sentiment=='negative'].iloc[:len(data_p)]
    data= pd.concat([data_n,data_p])
    # data['review']=(data.airline_sentiment=='positive').astype('int')
    # del data['review']
    data['review']=(data.airline_sentiment=='positive').astype('int')
    print(data)
    del data['airline_sentiment']
    token = re.compile('[A-Za-z]+|[!?,.()]')
    def reg_text(text):
        new_text=token.findall(text)
        new_text=[word.lower() for word in new_text]
        return new_text

    data['text'] = data.text.apply(reg_text)

    word_set=set()
    for text in data.text:
        for word in text:
            word_set.add(word)
    # print(len(word_set))
    # 将0 保留作为填充
    word_index=dict((key,value+1) for value,key in enumerate(list(word_set)))
    # print(word_index)
    data_ok=data.text.apply(lambda x: [ word_index[word] for word in x])
    # max_word 为所有文章中单词的种类，maxlen 为每个文章由多少单词 max_word+1 0作为补位填充 因此总单词量要+1
    max_word = len(word_set)+1
    maxlen = max(len(x) for x in data_ok)
    # 此处要将dataframe 或 series 类型的数据通过values方法转化成 ndarray数组在使用 keras 的方法
    data_ok=keras.preprocessing.sequence.pad_sequences(data_ok.values,maxlen) # 将所有样本填充到maxlen长度
    model=keras.Sequential()
    model.add(keras.layers.Embedding(max_word, 50, input_length=maxlen))
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
    # 另一种方式对数据进行划分测试和训练和batch validation_split测试集比例，此时data为ndarray values 表示转化为 ndarray数组
    history=model.fit(data_ok,data.review.values,epochs=10,batch_size=128,validation_split=0.2)
    print(history)



def Pm25_predict():
    data = pd.read_csv('./Data/PRSA_data_2010.1.1-2014.12.31.csv')
    data = data.iloc[24:].copy() #前24列 pm2.5为空
    # print(data['pm2.5'].isna())
    data.fillna(method='ffill',inplace=True) # ffill 表示勇用前一项填充当前的 NaN 值 后一项用
    data.drop('No', axis=1, inplace=True) # 删除 No列
    data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'],
                                                          month=x['month'],
                                                          day=x['day'],
                                                          hour=x['hour']),
                              axis=1)
    data.drop(columns=['year','month','day','hour'],inplace=True)
    data.set_index('time', inplace=True) # 将time列设置为索引
    # print(data)
    data = data.join(pd.get_dummies(data.cbwd)) # 将 cbwd 进行one-hot编码
    del data['cbwd']
    # print(data.head())
    sequence_length = 5 * 24
    delay = 24
    data_ = []
    for i in range(len(data) - sequence_length - delay):
        data_.append(data.iloc[i: i + sequence_length + delay])
    data_ = np.array([df.values for df in data_]) # 将dataframe转化为 array（numpy）
    np.random.shuffle(data_)
    x = data_[:, :-delay, :]
    y = data_[:, -1, 0]
    split_boundary = int(data_.shape[0] * 0.8)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    print(train_x)
    mean = train_x.mean(axis=0) # axis=0 对列操作 （特征）
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std #训练数据测试数据都是用训练数据的mean std做标准化，一般标签不做标准化，因为特征需要输入网络，预测值不需要输入网络

    model = keras.Sequential()
    model.add(layers.LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True)) # input_shape 不包含样本数量 堆叠lstm要使用return_sequences参数（返回序列中每个数据对应的输出 ，作为后边lstm的输入）
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(32)) #最后一个lstm 不需要 return_sequences=True
    model.add(layers.Dense(1))


    learning_rate_reduction=keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.0001) # 使用回调函数调整学习速率，当val_loss连续三次不降低时，将学习率*0.5，最小降低到0.0001
    model.compile(optimizer=keras.optimizers.Adam(), loss='mae')
    history = model.fit(train_x, train_y,
                        batch_size = 128,
                        epochs=200,
                        validation_data=(test_x, test_y),
                        callbacks=[learning_rate_reduction]
                        )
    print(history)

if __name__ == '__main__':
    # RNN_demo()
    Pm25_predict()