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

        # 记录自定义变量
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


# 自定义微分函数 使用 tensorboard 可视化网络训练参数
def tensorboard_test():

    # 自定义损失和优化函数
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizers = tf.optimizers.Adam()  # 优化器实例化
    train_loss =tf.metrics.Mean('train_loss')
    train_accuracy =tf.metrics.SparseCategoricalAccuracy('train_acc')
    test_loss =tf.metrics.Mean('test_loss')
    test_accuracy =tf.metrics.SparseCategoricalAccuracy('test_acc')

    train_log_dir_str = 'log/gradient_tape' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')+'/train'
    test_log_dir_str = 'log/gradient_tape' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')+'/test'
    train_file_writer = tf.summary.create_file_writer(train_log_dir_str)     #实例化文件编写器
    test_file_writer = tf.summary.create_file_writer(test_log_dir_str)     #实例化文件编写器


    # 计算损失函数值
    def loss(model, x, y):
        y_ = model(x)
        return loss_func(y, y_)
        # return keras.losses.SparseCategoricalCrossentropy(y,y_)

    # 对每个 batch 进行训练
    def train_step(model, images, labels):
        with tf.GradientTape() as t:
            pred = model(images,labels)
            loss_step = loss_func(labels,pred)
        grad = t.gradient(loss_step, model.trainable_variables)
        optimizers.apply_gradients(zip(grad, model.trainable_variables))
        train_loss(loss_step)
        train_accuracy(labels,pred)

    # 训练
    def train(model, dataset,test_ds):
        for epoch in range(10):
            for (batch, (images, labels)) in enumerate(dataset):  # 每次迭代遍历所有 batch()
                train_step(model, images, labels)
            with train_file_writer.as_default():
                tf.summary.scalar('loss',train_loss.result(),step=epoch)
                tf.summary.scalar('accuracy',train_accuracy.result(),step=epoch)
            print('epoch{} loss is {}, accuracy is {}'.format(epoch,train_loss.result(),train_accuracy.result()))
            train_loss.reset_states()
            train_accuracy.reset_states()
            for (batch, (images, labels)) in enumerate(test_ds):
                pred=model(images)
                test_loss(loss_func(labels,pred))
                test_accuracy(labels,pred)
            with test_file_writer.as_default():
                tf.summary.scalar('val_loss',test_loss.result(),step=epoch)
                tf.summary.scalar('accuracy',test_accuracy.result(),step=epoch)
            print('epoch{} testloss is {}, testaccuracy is {}'.format(epoch, test_loss.result(), test_accuracy.result()))
            test_loss.reset_states()
            test_accuracy.reset_states()


    # 加载数据集
    (train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()

    # 扩充维度，转换数据类型，归一化
    train_image = tf.expand_dims(train_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    train_image = tf.cast(train_image / 255, tf.float32)
    train_labels = tf.cast(train_labels, tf.int64)

    # 将数据转化为 tensor
    ds = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
    ds = ds.shuffle(1000).batch(32)
    # 测试数据
    # 扩充维度，转换数据类型，归一化
    test_image = tf.expand_dims(test_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    test_image = tf.cast(test_image / 255, tf.float32)
    test_labels = tf.cast(test_labels, tf.int64)

    # 将数据转化为 tensor
    test_ds = tf.data.Dataset.from_tensor_slices((test_image, test_labels))

    # 数据乱序 分batch
    test_ds = test_ds.batch(32)

    # 创建模型
    model = keras.Sequential()
    model.add(layers.Conv2D(16, [3, 3], activation='relu', input_shape=(28, 28, 1)))  #
    model.add(layers.Conv2D(32, [3, 3], activation='relu'))  #
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(10))



    # train(model, ds)
    train(model, ds,test_ds)


    # 启动 tensorboard
    # 在命令行输入以下代码
    # tensorboard -logdir log目录



if __name__ == '__main__':
    tensorboard_test()
