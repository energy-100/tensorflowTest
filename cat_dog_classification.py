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
    # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作
    label=tf.reshape(label,[1])
    return image,label

def cat_dog_classification():
    train_image_path=glob.glob('./data/dc_2000/train/*/*.jpg')
    test_image_path=glob.glob('./data/dc_2000/test/*/*.jpg')
    train_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in train_image_path]
    test_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in test_image_path]
    train_ds=tf.data.Dataset.from_tensor_slices((train_image_path,train_image_label))
    train_ds.map(load_process_file,num_parallel_calls=tf.data.experimental.AUTOTUNE) #使用多线程，线程数自适应
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    BATCH_SIZE=32
    train_count=len(train_image_path)
    test_count=len(test_image_path)
    train_ds=train_ds.repeat().shuffle(train_count).batch(BATCH_SIZE)
    test_ds=test_ds.batch(BATCH_SIZE)
    train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds=test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    imgs,labels =next(iter(train_ds))
    pass



if __name__ == '__main__':
    cat_dog_classification()