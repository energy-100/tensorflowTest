import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import glob
import os
import matplotlib.pyplot as plt


def load_process_file(filepath,label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[256,256])
    image=tf.cast(image,tf.float32)/255
    image=image/255
    # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作
    label=tf.reshape(label,[1])
    return image,label

# 猫狗数据集 自定义训练
def cat_dog_classification():

    # 数据预处理
    train_image_path=glob.glob('./data/dc_2000/test/*/*.jpg')
    test_image_path=glob.glob('./data/dc_2000/test/*/*.jpg')

    train_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in train_image_path]
    test_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in test_image_path]
    train_ds=tf.data.Dataset.from_tensor_slices((train_image_path,train_image_label))
    train_ds=train_ds.map(load_process_file,num_parallel_calls=tf.data.experimental.AUTOTUNE) #使用多线程，线程数自适应
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    test_ds=test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    BATCH_SIZE=32
    train_count=len(train_image_path)
    test_count=len(test_image_path)
    train_ds=train_ds.shuffle(train_count).batch(BATCH_SIZE)
    test_ds=test_ds.batch(BATCH_SIZE)
    train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds=test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # imgs,labels =next(iter(train_ds))

    # 构建模型
    model=keras.Sequential([
        keras.layers.Conv2D(64,[3,3],input_shape=[256,256,3],activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(128,[3,3],activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(256, [3, 3], activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(512, [3, 3], activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(1024, [3, 3], activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256,activation='relu'), #二分类问题
        keras.layers.Dense(1)  #二分类问题
    ])
    # print(model.summary())
    # imgs, labels = next(iter(train_ds))
    # pred=model(imgs)
    # print(pred)
    # print(np.array([ int(p[0].numpy()>0) for p in pred ]))

    # 定义所需对象
    # 大写返回小写的函数
    ls = keras.losses.BinaryCrossentropy(from_logits=True)
    optimiter = keras.optimizers.Adam(0.001)
    epoch_loss = tf.keras.metrics.Mean('train_loss')
    epoch_accuracy = tf.keras.metrics.Accuracy('train_accuracy')

    # 定义自定义函数
    def train_step(model,images,labels):
        with tf.GradientTape() as t:
            pred =model(images)
            step_loss = keras.losses.BinaryCrossentropy(from_logits=True)(labels,pred)
        grads = t.gradient(step_loss,model.trainable_variables)
        optimiter.apply_gradients(zip(grads,model.trainable_variables))
        epoch_loss(step_loss)
        # print(tf.cast(pred > 0, tf.int32).numpy())
        epoch_accuracy(labels,tf.cast(pred>0,tf.int32))



    def train(model,train_ds,test_ds):
        train_loss_result = []
        train_accuracy_result = []
        num_epoch = 10
        for epoch in range(num_epoch):
            for batch,(images,labels) in enumerate(train_ds):
                train_step(model,images,labels)
                print('.',end='')
            print()
            train_loss_result.append(epoch_loss.result())
            train_accuracy_result.append(epoch_accuracy.result())
            print('epoch {}, loss={:.3f}, accuracy={:.3f}'.format(epoch+1,epoch_loss.result(),epoch_accuracy.result()))
            epoch_loss.reset_states()
            epoch_accuracy.reset_states()

    train(model,train_ds,test_ds)


if __name__ == '__main__':
    cat_dog_classification()