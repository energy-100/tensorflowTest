import tensorflow as tf
import numpy as np

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
if __name__ == "__main__":
    # key_hot_test()
    demo1()