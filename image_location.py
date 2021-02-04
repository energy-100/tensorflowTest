from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import random
from lxml import etree
import numpy
from matplotlib.patches import Rectangle
def image_location_demo():

    image=glob.glob('./Data/location_image/images/*.jpg')
    xmls=glob.glob('./Data/location_image/annotations/xmls/*.xml')

    names=[path.split('\\')[-1].split('.')[0] for path in xmls]
    image=[img for img in image if img.split('\\')[-1].split('.')[0] in names]
    image2=[img for img in image if img.split('\\')[-1].split('.')[0] not in names]
    print(len(image))

    # 排序 将xmls和image_train对应
    image.sort(key=lambda x:x.split('\\')[-1].split('.')[0])
    xmls.sort(key=lambda x:x.split('\\')[-1].split('.')[0])
    # print(image_train[-5:])
    # print(xmls[-5:])

    # 返回每个图片目标区域的四个关键点的比例
    def to_label(path):
        xml=open(path).read()
        sel=etree.HTML(xml)
        width=int(sel.xpath('//size/width/text()')[0])
        heigh=int(sel.xpath('//size/height/text()')[0])
        xmin=int(sel.xpath('//bndbox/xmin/text()')[0])
        xmax=int(sel.xpath('//bndbox/xmax/text()')[0])
        ymin=int(sel.xpath('//bndbox/ymin/text()')[0])
        ymax=int(sel.xpath('//bndbox/ymax/text()')[0])
        return [xmin/width,xmax/width,ymin/heigh,ymax/heigh]

    label=[to_label(e) for e in xmls]
    out1,out2,out3,out4=list(zip(*label)) # list(zip(*label))返回的的是一个列表，列表的每一个元素是一个元祖

    # 构造标签
    # zip返回的元素返回的是元组，而from_tensor_slices()函数的参数是list或ndarray，需要进行转换
    # zip返回的是一个生成器 需要用list进行转换
    label_dataset=tf.data.Dataset.from_tensor_slices((list(out1),list(out2),list(out3),list(out4)))

    def load_image(path):
        image=tf.io.read_file(path)
        image=tf.image.decode_jpeg(image,channels=3)
        image=tf.image.resize(image,[244,244])
        image=tf.cast(image,tf.float32)
        image=image/127.5-1
        return image
    # 构造样本
    image_ds=tf.data.Dataset.from_tensor_slices(image)
    image_ds=image_ds.map(load_image)

    dataset=tf.data.Dataset.zip((image_ds,label_dataset))
    test_count = int(len(image) * 0.2)
    train_count = len(image) - test_count

    dataset_train = dataset.skip(test_count)
    dataset_test = dataset.take(test_count)
    BATCH_SIZE = 8
    BUFFER_SIZE = 300
    dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #要对自身赋值
    dataset_test = dataset_test.batch(BATCH_SIZE) #要对自身赋值

    # print(dataset)
    # for img,label in dataset.take(1):
    #     print(img[0])
    #     xmin,xmax,ymin,ymax=label[0][0].numpy()*244,label[1][0].numpy()*244,label[2][0].numpy()*244,label[3][0].numpy()*244 # 第一个维度代表四个参数编号 第二个代表batch 要乘以统一后的长或宽（四个参数是比例）
    #     tmp=tf.keras.preprocessing.image.array_to_img(img[0])
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    #     rect=Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,fill=False,color='red')
    #     ax=plt.gca()
    #     ax.axes.add_patch(rect)
    #     plt.show()


    # 创建模型
    xception = keras.applications.Xception(weights='imagenet',include_top=False,input_shape=(224,224,3))
    inputs=tf.keras.layers.Input(shape=(224,224,3))
    x=xception(inputs=inputs)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=keras.layers.Dense(2048,activation='relu')(x)
    x=keras.layers.Dense(256,activation='relu')(x)
    out1=keras.layers.Dense(1,activation='relu')(x)
    out2=keras.layers.Dense(1,activation='relu')(x)
    out3=keras.layers.Dense(1,activation='relu')(x)
    out4=keras.layers.Dense(1,activation='relu')(x)
    prediction=[out1,out2,out3,out4]
    model=keras.models.Model(inputs=inputs,outputs=prediction)
    model.compile(loss=keras.losses.MSE,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['mae'])

    STEPS_PER_EPOCH = train_count // BATCH_SIZE
    VALIDATION_STEPS = test_count // BATCH_SIZE
    hsitory = model.fit(dataset_train,validation_data=dataset_test,validation_steps=VALIDATION_STEPS,epochs=10,steps_per_epoch=STEPS_PER_EPOCH)
    print(hsitory)
    model.save('./model_save_file')

def show_predict():
    image=glob.glob('./Data/location_image/images/*.jpg')
    xmls=glob.glob('./Data/location_image/annotations/xmls/*.xml')

    names=[path.split('\\')[-1].split('.')[0] for path in xmls]
    image=[img for img in image if img.split('\\')[-1].split('.')[0] in names]
    image2=[img for img in image if img.split('\\')[-1].split('.')[0] not in names]
    print(len(image))

    # 排序 将xmls和image_train对应
    image.sort(key=lambda x:x.split('\\')[-1].split('.')[0])
    xmls.sort(key=lambda x:x.split('\\')[-1].split('.')[0])
    # print(image_train[-5:])
    # print(xmls[-5:])

    # 返回每个图片目标区域的四个关键点的比例
    def to_label(path):
        xml=open(path).read()
        sel=etree.HTML(xml)
        width=int(sel.xpath('//size/width/text()')[0])
        heigh=int(sel.xpath('//size/height/text()')[0])
        xmin=int(sel.xpath('//bndbox/xmin/text()')[0])
        xmax=int(sel.xpath('//bndbox/xmax/text()')[0])
        ymin=int(sel.xpath('//bndbox/ymin/text()')[0])
        ymax=int(sel.xpath('//bndbox/ymax/text()')[0])
        return [xmin/width,xmax/width,ymin/heigh,ymax/heigh]

    label=[to_label(e) for e in xmls]
    out1,out2,out3,out4=list(zip(*label)) # list(zip(*label))返回的的是一个列表，列表的每一个元素是一个元祖

    # 构造标签
    # zip返回的元素返回的是元组，而from_tensor_slices()函数的参数是list或ndarray，需要进行转换
    # zip返回的是一个生成器 需要用list进行转换
    label_dataset=tf.data.Dataset.from_tensor_slices((list(out1),list(out2),list(out3),list(out4)))

    def load_image(path):
        image=tf.io.read_file(path)
        image=tf.image.decode_jpeg(image,channels=3)
        image=tf.image.resize(image,[244,244])
        image=tf.cast(image,tf.float32)
        image=image/127.5-1
        return image
    # 构造样本
    image_ds=tf.data.Dataset.from_tensor_slices(image)
    image_ds=image_ds.map(load_image)

    dataset=tf.data.Dataset.zip((image_ds,label_dataset))
    test_count = int(len(image) * 0.2)

    dataset_train = dataset.skip(test_count)
    dataset_test = dataset.take(test_count)
    BATCH_SIZE = 8
    BUFFER_SIZE = 300
    dataset_train = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 要对自身赋值

    new_model=tf.keras.models.load_model('./model_save_file')
    plt.figure(figsize=(8,24))
    for img,_ in dataset_test.take(1):
        out1,out2,out3,out4 =new_model.predict(img)
        for i in range(3):
            plt.subplot(3,1,i+1)
            xmin,xmax,ymin,ymax=out1[i]*244,out2[i]*244,out3[i]*244,out4[i]*244
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red')
            ax=plt.gca()
            ax.axes.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    image_location_demo()