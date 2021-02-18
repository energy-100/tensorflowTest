# import tensorflow as tf


# image = tf.io.read_file('Data/birds/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg')
# print(image)

# dic={'a':1,'b':2,'c':3}
# string='{a}-{b}-{c}'.format(**dic)
# aa={'a':1,'b':2,'c':3}
# print(*aa)
# c=**dic
# print(**aa)
# def fun(a, *args):
#      # for e in args:
#      #     print(e)
#      print(args)
#
# fun("Geek", "dog", "cat")
# print(list(zip([1,1,1,1,1,1],[2,2,2,2,2,2])))

#导包
import pandas as pd
import numpy as np
# from numpy import nan as NaN
# df1=pd.DataFrame([[1,2,3],[NaN,NaN,2],[NaN,NaN,NaN],[8,8,NaN]])

# print (df1.fillna(100))
# print ("-----------------------")
# print(df1)
# print(df1.fillna({0:10,1:20,2:30}))
# print (df1.fillna(0,inplace=True))
# print ("-------------------------")
# print (df1)
# df2 = pd.DataFrame(np.random.randint(0,10,(5,5)))
# # df2.iloc[1:4,3] = None
# # df2.iloc[2:4,4] = None
# # print(df2)
# # print ("-------------------------")
# # print(df2.fillna(method='ffill'))
df2 = pd.DataFrame(np.random.randint(0,10,(5,5)))
df2.iloc[1:4,3] = None
df2.iloc[2:4,4] = None
print(df2.fillna(method="ffill", limit=1, axis=1))