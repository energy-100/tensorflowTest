from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import random


def MultoutModel_demo():
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = image / 255.0  # normalize to [0,1] range
        image = 2 * image - 1 #归一化 -1~1
        print(image)
        return image



    all_image_paths=glob.glob('Data/moc/*/*')
    print(all_image_paths)
    # print(len(all_image_paths))
    random.shuffle(all_image_paths)
    label_names=set(elem.split('.')[-2].split('\\')[-2] for elem in all_image_paths)
    color_label_names = set(name.split('_')[0] for name in label_names)
    item_label_names = set(name.split('_')[1] for name in label_names)

    color_label_to_index=dict((key,value) for value,key  in enumerate(color_label_names))
    item_label_names_to_index=dict((key,value) for value,key in enumerate(item_label_names))

    color_label=[color_label_to_index[elem.split('.')[-2].split('\\')[-2].split('_')[0]] for elem in all_image_paths]
    item_label=[item_label_names_to_index[elem.split('.')[-2].split('\\')[-2].split('_')[1]] for elem in all_image_paths]

    train_count=int(len(all_image_paths)*0.8)
    train_data_ds=tf.data.Dataset.from_tensor_slices(all_image_paths[:train_count])
    test_data_ds=tf.data.Dataset.from_tensor_slices(all_image_paths[train_count:])
    # 注意，此处map函数要重新赋值
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data_ds=train_data_ds.map(load_and_preprocess_image, num_parallel_calls = AUTOTUNE)
    test_data_ds=test_data_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    train_label_ds=tf.data.Dataset.from_tensor_slices((color_label[:train_count],item_label[:train_count]))
    test_label_ds=tf.data.Dataset.from_tensor_slices((color_label[train_count:],item_label[train_count:]))

    train_ds=tf.data.Dataset.zip((train_data_ds,train_label_ds))
    test_ds=tf.data.Dataset.zip((test_data_ds,test_label_ds))

    BATCH_SIZE=32
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    train_ds = train_ds.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(BATCH_SIZE)

    # 此模型没有使用预训练权重，只使用了网络架构
    model_net=keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,pooling='avg')
    input=tf.keras.Input(shape=(224,224,3))
    x=model_net(input)
    x1=tf.keras.layers.Dense(1024,activation='relu')(x)
    out_color=tf.keras.layers.Dense(len(color_label_names),activation='softmax',name='out_color')(x1)
    x2=tf.keras.layers.Dense(1024,activation='relu')(x)
    out_item=tf.keras.layers.Dense(len(item_label_names),activation='softmax',name='out_item')(x2)
    model=tf.keras.Model(inputs=input,outputs=[out_color,out_item])
    print(model.summary())

    # 此处loss 和 optimizer 用大写 表示返回函数的函数 小写一般用在自定义函数中（需要输入参数）
    # 非 one-hot 编码要使用Sparse开头的损失函数
    # 如果多个输出的损失函数相同写一个就行 如果有两个不同则要对loss字段传入字典（和模型输出的名字相对应）
    # 损失函数和优化函数如果使用字符串 要小写
    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  # loss={'out_color',keras.losses.SparseCategoricalCrossentropy(),'out_item',keras.losses.SparseCategoricalCrossentropy()},
                  metrics=['acc'])
    history=model.fit(train_ds,validation_data=test_ds,epochs=1,steps_per_epoch=train_count//BATCH_SIZE,validation_steps=(len(all_image_paths)-train_count)//BATCH_SIZE)
    print(history)
    # 评价模型
    print(model.evaluate(test_data_ds,test_label_ds,verbose=1)) # 第一项是损失，第二项是准确率
    predict_image=load_and_preprocess_image(r"D:\PyCharm_GitHub_local_Repository\tensorflowTest\Data\moc\black_jeans\00000000.jpg")
    # 扩充维度
    predict_image=np.expand_dims(predict_image,0)
    pre=model.predict(predict_image)
    precolor=np.argmax(pre[0][0]) #样本号，预测标签序号
    print(precolor)

    # 使用函数API对模型进行预测
    pre=model(predict_image,training=True)

def save_model():
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = image / 255.0  # normalize to [0,1] range
        image = 2 * image - 1 #归一化 -1~1
        print(image)
        return image



    all_image_paths=glob.glob('Data/moc/*/*')
    print(all_image_paths)
    # print(len(all_image_paths))
    random.shuffle(all_image_paths)
    label_names=set(elem.split('.')[-2].split('\\')[-2] for elem in all_image_paths)
    color_label_names = set(name.split('_')[0] for name in label_names)
    item_label_names = set(name.split('_')[1] for name in label_names)

    color_label_to_index=dict((key,value) for value,key  in enumerate(color_label_names))
    item_label_names_to_index=dict((key,value) for value,key in enumerate(item_label_names))

    color_label=[color_label_to_index[elem.split('.')[-2].split('\\')[-2].split('_')[0]] for elem in all_image_paths]
    item_label=[item_label_names_to_index[elem.split('.')[-2].split('\\')[-2].split('_')[1]] for elem in all_image_paths]

    train_count=int(len(all_image_paths)*0.8)
    train_data_ds=tf.data.Dataset.from_tensor_slices(all_image_paths[:train_count])
    test_data_ds=tf.data.Dataset.from_tensor_slices(all_image_paths[train_count:])
    # 注意，此处map函数要重新赋值
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_data_ds=train_data_ds.map(load_and_preprocess_image, num_parallel_calls = AUTOTUNE)
    test_data_ds=test_data_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    train_label_ds=tf.data.Dataset.from_tensor_slices((color_label[:train_count],item_label[:train_count]))
    test_label_ds=tf.data.Dataset.from_tensor_slices((color_label[train_count:],item_label[train_count:]))

    train_ds=tf.data.Dataset.zip((train_data_ds,train_label_ds))
    test_ds=tf.data.Dataset.zip((test_data_ds,test_label_ds))

    BATCH_SIZE=16
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    train_ds = train_ds.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(BATCH_SIZE)

    # 此模型没有使用预训练权重，只使用了网络架构
    model_net=keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,pooling='avg')
    input=tf.keras.Input(shape=(224,224,3))
    x=model_net(input)
    x1=tf.keras.layers.Dense(1024,activation='relu')(x)
    out_color=tf.keras.layers.Dense(len(color_label_names),activation='softmax',name='out_color')(x1)
    x2=tf.keras.layers.Dense(1024,activation='relu')(x)
    out_item=tf.keras.layers.Dense(len(item_label_names),activation='softmax',name='out_item')(x2)
    model=tf.keras.Model(inputs=input,outputs=[out_color,out_item])
    print(model.summary())

    # 此处loss 和 optimizer 用大写 表示返回函数的函数 小写一般用在自定义函数中（需要输入参数）
    # 非 one-hot 编码要使用Sparse开头的损失函数
    # 如果多个输出的损失函数相同写一个就行 如果有两个不同则要对loss字段传入字典（和模型输出的名字相对应）
    # 损失函数和优化函数如果使用字符串 要小写
    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  # loss={'out_color',keras.losses.SparseCategoricalCrossentropy(),'out_item',keras.losses.SparseCategoricalCrossentropy()},
                  metrics=['acc'])

    # 使用回调函数保存模型
    checkpoint_path = './model_save_file/checkpoint.ckpt'
    cpcallback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False)

    history=model.fit(train_ds,
                      validation_data=test_ds,
                      epochs=2,steps_per_epoch=train_count//BATCH_SIZE,
                      validation_steps=(len(all_image_paths)-train_count)//BATCH_SIZE,
                      callbacks=[cpcallback])
    print(history)
    # 评价模型
    # model.evaluate 用于评估您训练的模型。它的输出是准确度或损失，而不是对输入数据的预测。
    # model.predict 实际预测，其输出是目标值，根据输入数据预测。
    model.evaluate(test_ds,verbose=1)
    # model.evaluate(train_ds,test_ds,verbose=1) 错误写法 输入是 dataset时 样本和标签是一体的
    predict_image=load_and_preprocess_image(r"D:\PyCharm_GitHub_local_Repository\tensorflowTest\Data\moc\black_jeans\00000000.jpg")
    # 扩充维度
    predict_image=np.expand_dims(predict_image,0)
    pre=model.predict(predict_image)
    precolor=np.argmax(pre[0][0]) #样本号，预测标签序号
    print(precolor)

    # 使用函数API对模型进行预测
    pre=model(predict_image,training=True)

    # 保存整个模型
    model.save('model_save_file/saveModelDemo.h5') # 保存模型所有数据
    new_model=tf.keras.models.load_model('model_save_file/saveModelDemo.h5') # 加载模型全部数据

    # 保存模型结构
    json_model=model.to_json() # 保存模型结构保存到json
    new_model=tf.keras.models.model_from_json(json_model) # 从json文件中读取模型结构

    # 保存权重
    weight=model.get_weights() #获取权重
    model.set_weights(weight)   #设置权重
    model.save_weights('model_save_file/weight_file.h5') # 保存权重到文件
    model.load_weights('model_save_file/weight_file.h5') # 从文件读取权重
    print(new_model.summary())



if __name__ == '__main__':
    # MultoutModel_demo()
    save_model()