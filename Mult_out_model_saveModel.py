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


def customize_save_model():
    def load_process_file(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [360, 360])
        image = tf.image.random_crop(image, [256, 256, 3])  # 随机裁剪
        image = tf.image.random_flip_left_right(image)  # 随机左右反转
        image = tf.image.random_flip_up_down(image)  # 随机上下反转
        image = tf.image.random_brightness(image, 0.5)  # 随机改变亮度
        image = tf.cast(image, tf.float32) / 255
        image = image / 255
        # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作
        label = tf.reshape(label, [1])
        return image, label

    # 猫狗数据集 自定义训练
    def cat_dog_classification():

        # 数据预处理
        train_image_path = glob.glob('./data/mini_cat_dog/train/*/*.jpg')
        test_image_path = glob.glob('./data/mini_cat_dog/train/*/*.jpg')

        train_image_label = [1 if elem.split('\\')[1] == 'dog' else 0 for elem in train_image_path]
        test_image_label = [1 if elem.split('\\')[1] == 'dog' else 0 for elem in test_image_path]
        train_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
        train_ds = train_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
        test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
        test_ds = test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
        BATCH_SIZE = 32
        train_count = len(train_image_path)
        test_count = len(test_image_path)
        train_ds = train_ds.shuffle(train_count).batch(BATCH_SIZE)
        test_ds = test_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # imgs,labels =next(iter(train_ds))

        # 构建模型
        model = keras.Sequential([
            keras.layers.Conv2D(64, [3, 3], input_shape=[256, 256, 3], activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(128, [3, 3], activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(256, [3, 3], activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(512, [3, 3], activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(1024, [3, 3], activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),  # 二分类问题
            keras.layers.Dense(1)  # 二分类问题
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
        def train_step(model, images, labels):
            with tf.GradientTape() as t:
                pred = model(images)
                step_loss = keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
            grads = t.gradient(step_loss, model.trainable_variables)
            optimiter.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss(step_loss)
            # print(tf.cast(pred > 0, tf.int32).numpy())
            epoch_accuracy(labels, tf.cast(pred > 0, tf.int32))

        def train(model, train_ds, test_ds):
            train_loss_result = []
            train_accuracy_result = []
            num_epoch = 10
            for epoch in range(num_epoch):
                for batch, (images, labels) in enumerate(train_ds):
                    train_step(model, images, labels)
                    print('.', end='')
                print()
                train_loss_result.append(epoch_loss.result())
                train_accuracy_result.append(epoch_accuracy.result())
                print('epoch {}, loss={:.3f}, accuracy={:.3f}'.format(epoch + 1, epoch_loss.result(),
                                                                      epoch_accuracy.result()))
                epoch_loss.reset_states()
                epoch_accuracy.reset_states()

        train(model, train_ds, test_ds)

    def kaggle_run_test():
        # 自有代码测试
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
        np.random.shuffle(train_image_path)
        train_image_label = [1 if elem.split('/')[5] == 'dogs' else 0 for elem in train_image_path]

        train_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
        train_ds = train_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
        #     test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
        #     test_ds=test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
        BATCH_SIZE = 32
        train_count = len(train_image_path)
        test_count = len(test_image_path)
        train_ds = train_ds.shuffle(train_count).batch(BATCH_SIZE)
        #     test_ds=test_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        #     test_ds=test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # imgs,labels =next(iter(train_ds))

        # 构建模型
        model = keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1)
        ])
        # print(model.summary())
        # imgs, labels = next(iter(train_ds))
        # pred=model(imgs)
        # print(pred)
        # print(np.array([ int(p[0].numpy()>0) for p in pred ]))

        # 定义所需对象
        # 大写返回小写的函数
        ls = keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        # tf.keras.optimizers.Adam(learning_rate=0.0001)
        epoch_loss = tf.keras.metrics.Mean('train_loss')
        epoch_accuracy = tf.keras.metrics.Accuracy('train_accuracy')
        checkpoint=tf.train.Checkpoint(optimizer=optimizer,
                                       model=model)
        cp_dir = './customtrain_cp'
        cp_prefix=cp_dir+"/ckpt"  #设置一个文件名的前缀

        # 定义自定义函数
        def train_step(model, images, labels):
            with tf.GradientTape() as t:
                pred = model(images)
                step_loss = keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
            grads = t.gradient(step_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss(step_loss)
            # print(tf.cast(pred > 0, tf.int32).numpy())
            epoch_accuracy(labels, tf.cast(pred > 0, tf.int32))

        # def train_step(model, images, labels):
        #     with tf.GradientTape() as t:
        #         pred = model(images)
        #         loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
        #     grads = t.gradient(loss_step, model.trainable_variables)
        #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #     epoch_loss(loss_step)
        #     epoch_accuracy(labels, tf.cast(pred > 0, tf.int32))

        def train(model, train_ds):
            train_loss_result = []
            train_accuracy_result = []
            num_epoch = 30
            for epoch in range(num_epoch):
                for batch, (images, labels) in enumerate(train_ds):
                    train_step(model, images, labels)
                    print('.', end='')
                print()
                train_loss_result.append(epoch_loss.result())
                train_accuracy_result.append(epoch_accuracy.result())
                print('epoch {}, loss={:.3f}, accuracy={:.3f}'.format(epoch + 1, epoch_loss.result(),
                                                                      epoch_accuracy.result()))
                epoch_loss.reset_states()
                epoch_accuracy.reset_states()
                if epoch%2==0:
                    checkpoint.save(file_prefix=cp_prefix)

        train(model, train_ds)


        # 加载自定义训练模型
        cp_dir='./model_save_file'
        latest_checkpoint=tf.train.latest_checkpoint(cp_dir) # 获取最新检查点的信息
        checkpoint.restore(latest_checkpoint) # 此代码能够将保存的参数复制给网络



if __name__ == '__main__':
    # MultoutModel_demo()
    save_model()