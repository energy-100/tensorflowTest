import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime

def tensorboard_test():

    # 定义自适应学习率函数(自定义参数)
    def lr_sche(epoch):
        learning_rate=0.2
        if epoch>5:
            learning_rate=0.02
        if epoch>10:
            learning_rate = 0.01
        if epoch>20:
            learning_rate = 0.005

        #写入文件
        tf.summary.scalar('learning_rate',data=learning_rate,step=epoch)
        return learning_rate

    # 加载数据集
    (train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()

    # 扩充维度，转换数据类型，归一化
    train_image = tf.expand_dims(train_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    train_image = tf.cast(train_image / 255, tf.float32)
    train_labels = tf.cast(train_labels, tf.int64)

    # 将数据转化为 tensor
    ds = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
    ds = ds.repeat().shuffle(1000).batch(128)
    # 测试数据
    # 扩充维度，转换数据类型，归一化
    test_image = tf.expand_dims(test_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    test_image = tf.cast(test_image / 255, tf.float32)
    test_labels = tf.cast(test_labels, tf.int64)

    # 将数据转化为 tensor
    test_ds = tf.data.Dataset.from_tensor_slices((test_image, test_labels))

    # 数据乱序 分batch
    test_ds = test_ds.batch(128)

    # 构建模型
    model=tf.keras.Sequential()
    model.add(layers.Conv2D(16,[3,3],activation='relu',input_shape=(28,28,1)))
    model.add(layers.Conv2D(32,[3,3],activation='relu'))
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer=tf.optimizers.Adam(),loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    log_dir_str = 'log/'+datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_str,histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_sche)
    file_writer = tf.summary.create_file_writer(log_dir_str+'/lr')     #实例化文件编写器
    file_writer.set_as_default() # 设置为默认文件编写器


    model.fit(ds,validation_data=test_ds,
              epochs=10,steps_per_epoch=len(train_image)//128,
              validation_steps=len(test_image)//128,
              callbacks=[tensorboard_callback,lr_callback])

    # 启动 tensorboard
    # 在命令行输入以下代码
    # tensorboard -logdir log目录



if __name__ == '__main__':
    tensorboard_test()
