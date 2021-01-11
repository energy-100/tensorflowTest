import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
def demo1():
    # print(tf.__version__)
    # print(tf.executing_eagerly())
    x=[[2,]]
    m=tf.matmul(x,x)
    print(m)
    print(m.numpy())
    a=tf.constant([[1,2],[3,4]])
    b=tf.add(a,1)
    num=tf.convert_to_tensor(10)
    for i in range(num.numpy()):
        i=tf.constant(i)
        if int(i/2) ==0:
            print("a")
        else:
            print("b")
    d=np.array([[5,6],[7,8]])

# tensoflow eager模式下的变量运算
def eager_Variable():
    v=tf.Variable(10) #创建变量
    v.assign(5)  #改变变量
    v.assign_add(2)  #变量加法
    v.assign_sub(2)  #变量减法
    v1=v.read_value()  #读取变量的值，转换为numpy类型



# tensoflow eager模式下的上下文管理器 GradientTape()
# GradientTape()对于常量和变量的跟踪必须是float类型
# GradientTape()在调用一次求导之后便会释放资源，如果要多次求导 要使用 persistent=True 参数
def eager_tape_test():


    # 变量求微分 必须是float类型
    w=tf.Variable([[1.0]])
    # 建立上下文管理器
    with tf.GradientTape() as t:
        loss=w*w
    gard=t.gradient(loss,w)   #函数，自变量
    print(gard)


    # 常量求微分 必须是float类型
    w=tf.constant([3.0])
    # 建立上下文管理器
    with tf.GradientTape() as t:
        t.watch(w)  #对于常量要使用watch()方法手动跟踪
        loss=w*w
    gard=t.gradient(loss,w)   #函数，自变量
    print(gard)

    # 多次求导
    # 常量求微分 必须是float类型
    w = tf.Variable([[1.0]])
    # 建立上下文管理器
    with tf.GradientTape(persistent=True) as t:
        y=w*w
        z=y*y
    gard1=t.gradient(y,w)   #函数，自变量
    gard2=t.gradient(z,y)   #函数，自变量
    print(gard1)
    print(gard2)

# 损失函数 小写(函数或字符串)开头表示返回函数的函数，一般用在 complile 函数中,根据参数返回相应类型的小写的同名函数
# 小写表示函数需要添加参数执行，一般用在自定义损失函数和自定义优化函数中
def eager_tape_Demo():
    # 自定义损失和优化函数
    loss_func=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizers = tf.optimizers.Adam()  #优化器实例化

    # 计算损失函数值
    def loss(model,x,y):
        y_ =model(x)
        return loss_func(y,y_)
        # return keras.losses.SparseCategoricalCrossentropy(y,y_)

    # 对每个 batch 进行训练
    def train_step(model,images,labels):
        with tf.GradientTape() as t:
            loss_step = loss(model,images,labels)
        grad=t.gradient(loss_step,model.trainable_variables)
        optimizers.apply_gradients(zip(grad,model.trainable_variables))

    # 训练
    def train(model,dataset):
        for epoch in range(10):
            for (batch ,(images,labels)) in enumerate(dataset):  #每次迭代遍历所有 batch()
                train_step(model,images,labels)
            print('epoch{} is finished'.format(epoch))


    # 加载数据集
    (train_image,train_labels),_=tf.keras.datasets.mnist.load_data()

    # 扩充维度，转换数据类型，归一化
    train_image=tf.expand_dims(train_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    train_image=tf.cast(train_image/255, tf.float32)
    train_labels=tf.cast(train_labels,tf.int64)

    # 将数据转化为 tensor
    ds=tf.data.Dataset.from_tensor_slices((train_image,train_labels))

    # 数据乱序 分batch
    ds=ds.shuffle(10000).batch(32)

    # 创建模型
    model=keras.Sequential()
    model.add(layers.Conv2D(16,[3,3],activation='relu',input_shape=(28,28,1)))  #
    model.add(layers.Conv2D(32,[3,3],activation='relu'))  #
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(10))

    train(model,ds)


# 汇总计算模块 metrics 练习
def eager_metrics_test():

    # 均值demo
    m=tf.keras.metrics.Mean('acc')
    m(10)
    m(20)
    m([20,30])
    print(m.result().numpy())

    # 重置
    m.reset_states()
    m(1)
    m(2)
    m(3)
    print(m.result().numpy())

    # 准确率 demo

    # 实例化对象
    a=tf.keras.metrics.SparseCategoricalAccuracy('acc')

    # 加载数据集
    (train_image,train_labels),_=tf.keras.datasets.mnist.load_data()

    # 扩充维度，转换数据类型，归一化
    train_image=tf.expand_dims(train_image, -1)
    # train_labels=tf.expand_dims(train_labels, -1) #标签项无需修改
    train_image=tf.cast(train_image/255, tf.float32)
    train_labels=tf.cast(train_labels,tf.int64)

    # 将数据转化为 tensor
    ds=tf.data.Dataset.from_tensor_slices((train_image,train_labels))

    # 数据乱序 分batch
    ds=ds.shuffle(10000).batch(32)

    # 创建模型
    model=keras.Sequential()
    model.add(layers.Conv2D(16,[3,3],activation='relu',input_shape=(28,28,1)))  #
    model.add(layers.Conv2D(32,[3,3],activation='relu'))  #
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(10))

    features ,labels = next(iter(ds))
    predictions = model(features)
    print(a(labels,predictions).numpy()) #此方法会自动计算 predictions 中每组中的最大值和序号 并和labels比较


# 自定义微分编写
def eager_metrics_demo():
    # 自定义损失和优化函数
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizers = tf.optimizers.Adam()  # 优化器实例化
    train_loss =tf.metrics.Mean('train_loss')
    train_accuracy =tf.metrics.SparseCategoricalAccuracy('train_acc')
    test_loss =tf.metrics.Mean('test_loss')
    test_accuracy =tf.metrics.SparseCategoricalAccuracy('test_acc')

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
            print('epoch{} loss is {}, accuracy is {}'.format(epoch,train_loss.result(),train_accuracy.result()))
            train_loss.reset_states()
            train_accuracy.reset_states()
            for (batch, (images, labels)) in enumerate(test_ds):
                pred=model(images)
                test_loss(loss_func(labels,pred))
                test_accuracy(labels,pred)

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

if __name__ == "__main__":
    # key_hot_test()
    # demo1()
    # eager_Variable()
    # eager_tape_test()
    # eager_tape_Demo()
    # eager_metrics_test()
    eager_metrics_demo()