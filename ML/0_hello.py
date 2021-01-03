import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_URL="http://download.tensorflow.org/data/iris_training.csv"
train_path=tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

column_names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
df_iris=pd.read_csv(train_path,header=0,names=column_names)

iris=np.array(df_iris)

fig=plt.figure('Iris Data',figsize=(15,15))

plt.suptitle("Andreson's Iris Dara Set\n(Blue->Setosa|Red->Versicolor|Green->Virginical)")

for i in range(4):
    for j in range(4):
        plt.subplot(4,4,4*i+(j+1))
        if(i==j):
            plt.text(0.3,0.4,column_names[i],fontsize=15)
        else:
            plt.scatter(iris[:,j],iris[:,i],c=iris[:,4],cmap='brg')
            
        if(i==0):
            plt.title(column_names[j])
        if(j==0):
            plt.ylabel(column_names[i])

plt.show()