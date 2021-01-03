# 参考链接：https://blog.csdn.net/qq_42604176/article/details/108282705

"""
    TensorFlow实现一元逻辑回归
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([137.97,104.50,100.00,126.32,79.20,99.00,124.00,114.00,
    106.69,140.05,53.75,46.91,68.00,63.02,81.26,86.21])
y = np.array([1,1,0,1,0,1,1,0,0,1,0,0,0,0,0,0])
# plt.scatter(x,y)
# plt.show()
#plt.scatter(x,y)
#中心化操作，使得中心点为0
x_train=x-np.mean(x)
y_train=y
plt.scatter(x_train,y_train)
#设置超参数
learn_rate=0.005
#迭代次数
iter=5
#每10次迭代显示一下效果
display_step=1
#设置模型参数初始值
np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())
#观察初始参数模型
x_start=range(-80,80)
y_start=1/(1+tf.exp(-(w*x_start+b)))
plt.plot(x_start,y_start,color="red",linewidth=3)
#训练模型
#存放训练集的交叉熵损失、准确率
cross_train=[]  
acc_train=[]
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #sigmoid函数
        pred_train=1/(1+tf.exp(-(w*x_train+b)))
        #交叉熵损失函数
        Loss_train=-tf.reduce_mean(y_train*tf.math.log(pred_train)+(1-y_train)*tf.math.log(1-pred_train))
        #训练集准确率
        Accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train<0.5,0,1),y_train),tf.float32))
    #记录每一次迭代的损失和准确率
    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)    
    #更新参数
    dL_dw,dL_db = tape.gradient(Loss_train,[w,b])
    w.assign_sub(learn_rate*dL_dw)
    b.assign_sub(learn_rate*dL_db)
    #plt.plot(x,pred)
    if i % display_step==0:
        print("i:%i, Train Loss:%f,Accuracy:%f"%(i,cross_train[i],acc_train[i]))
        y_start=1/(1+tf.exp(-(w*x_start+b)))
        plt.plot(x_start,y_start)

#进行分类，并不是测试集，测试集是有标签的数据，而我们这边没有标签，这里是真实场景的应用情况
#商品房面积
x_test=[128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00,162.00,114.60]
#根据面积计算概率，这里使用训练数据的平均值进行中心化处理
pred_test=1/(1+tf.exp(-(w*(x_test-np.mean(x))+b)))
#根据概率进行分类
y_test=tf.where(pred_test<0.5,0,1) 
#打印数据
for i in range(len(x_test)):
    print(x_test[i],"\t",pred_test[i].numpy(),"\t",y_test[i].numpy(),"\t")
#可视化输出
plt.figure()
plt.scatter(x_test,y_test)
x_=np.array(range(-80,80))
y_=1/(1+tf.exp(-(w*x_+b)))
plt.plot(x_+np.mean(x),y_)
plt.show()