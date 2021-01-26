import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import glob
import numpy as np
def transfer_learning_kaggle():

    # 数据处理
    def load_process_file(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [360, 360])
        image = tf.cast(image, tf.float32) / 255
        image = image / 255
        # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作
        label = tf.reshape(label, [1])
        return image, label

    # 猫狗数据集 自定义训练

    # 数据预处理
    train_image_path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')
    test_image_path = glob.glob('../input/cat-and-dog/test_set/test_set/*/*.jpg')
    #     print(train_image_path)
    train_image_path = train_image_path[0:500]
    test_image_path = test_image_path[0:500]
    np.random.shuffle(train_image_path)
    train_image_label = [1 if elem.split('/')[5] == 'dogs' else 0 for elem in train_image_path]
    test_image_label = [1 if elem.split('/')[5] == 'dogs' else 0 for elem in test_image_path]

    train_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    train_ds = train_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    test_ds = test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    #     test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    #     test_ds=test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    BATCH_SIZE = 32
    train_count = len(train_image_path)
    test_count = len(test_image_path)
    train_ds = train_ds.shuffle(train_count).batch(BATCH_SIZE).repeat()
    test_ds=test_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


    # include_top=False z只引入卷积层的预训练权重
    covn_base = keras.applications.VGG16(weights='imagenet',include_top=False)
    print(covn_base.summary())

    #建立网络
    model=keras.Sequential()
    model.add(covn_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    # 将预训练网络部分设置为不训练权重
    covn_base.trainable=False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss=keras.losses.binary_crossentropy(),metrics=['accuracy'])
    model.fit(train_ds,test_ds,
              validation_data=test_ds,
              epoch=15,
              steps_per_epoch=train_count//BATCH_SIZE,
              validation_steps=test_count//BATCH_SIZE)

    # 微调 解冻顶部卷积层
    covn_base.trainable=True
    print(len(covn_base.layers))
    fine_tune_at = -3
    for layer in covn_base.layers[:fine_tune_at]:
        layer.trainable=False

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005/10),
                  loss=keras.losses.binary_crossentropy(),metrics=['accuracy'])
    initial_epoch=15 #设置epoch的开始位置
    fine_tune_epoch=10
    total_epoch=initial_epoch+fine_tune_epoch
    model.fit(train_ds,test_ds,
              validation_data=test_ds,
              epoch=total_epoch,
              initial_epoch=initial_epoch,
              steps_per_epoch=train_count//BATCH_SIZE,
              validation_steps=test_count//BATCH_SIZE)



# Xception 模型只支持高度，宽度，通道数结构的数据集
# 默认输入尺寸是 299*299
# include_top 是否包含全连接层
# weight None 代表随机初始化，'imagenet'表示加载在ImageNet上预训练的权值
# input_shape 可选，仅当 include_top=False时有效（否则形状必须是（299，299，3）因为预训练模型是以这个进行训练的）。他必须有三个通道，且宽高必须不小于71
# pooling 参数： 当 include_top=False 时 此参数有效 None代表不池化，直接将最后的卷积层输出（4D张量）
#   'avg'表示全局平均池化 在卷积层后添加一个全局平均池化层 输出2D张量（batch_size,feature_map_size）
#   'max'表示全局最大池化
# 详细信息参考 https://keras.io/zh/applications/


def transfer_learning_Xecption_kaggle():

    # 数据处理
    def load_process_file(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [360, 360])
        image = tf.cast(image, tf.float32) / 255
        image = image / 255
        # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作
        label = tf.reshape(label, [1])
        return image, label

    # 猫狗数据集 自定义训练

    # 数据预处理
    train_image_path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')
    test_image_path = glob.glob('../input/cat-and-dog/test_set/test_set/*/*.jpg')
    #     print(train_image_path)
    train_image_path = train_image_path[0:500]
    test_image_path = test_image_path[0:500]
    np.random.shuffle(train_image_path)
    train_image_label = [1 if elem.split('/')[5] == 'dogs' else 0 for elem in train_image_path]
    test_image_label = [1 if elem.split('/')[5] == 'dogs' else 0 for elem in test_image_path]

    train_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    train_ds = train_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    test_ds = test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    #     test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    #     test_ds=test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
    BATCH_SIZE = 32
    train_count = len(train_image_path)
    test_count = len(test_image_path)
    train_ds = train_ds.shuffle(train_count).batch(BATCH_SIZE).repeat()
    test_ds=test_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


    # include_top=False z只引入卷积层的预训练权重
    covn_base = keras.applications.xception.Xception(weights='imagenet',input_shape=(360,360,3),pooling='avg')
    print(covn_base.summary())

    #建立网络
    model=keras.Sequential()
    model.add(covn_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    # 将预训练网络部分设置为不训练权重
    covn_base.trainable=False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss=keras.losses.binary_crossentropy(),metrics=['accuracy'])
    model.fit(train_ds,test_ds,
              validation_data=test_ds,
              epoch=15,
              steps_per_epoch=train_count//BATCH_SIZE,
              validation_steps=test_count//BATCH_SIZE)

    # 微调 解冻顶部卷积层
    covn_base.trainable=True
    print(len(covn_base.layers))
    fine_tune_at = -33
    for layer in covn_base.layers[:fine_tune_at]:
        layer.trainable=False

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005/10),
                  loss=keras.losses.binary_crossentropy(),metrics=['accuracy'])
    initial_epoch=15 #设置epoch的开始位置
    fine_tune_epoch=10
    total_epoch=initial_epoch+fine_tune_epoch
    model.fit(train_ds,test_ds,
              validation_data=test_ds,
              epoch=total_epoch,
              initial_epoch=initial_epoch,
              steps_per_epoch=train_count//BATCH_SIZE,
              validation_steps=test_count//BATCH_SIZE)



if __name__ == '__main__':
    pass