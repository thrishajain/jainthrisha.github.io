import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('headbrain.csv')
x_train=numpy.array(data[:100,2])
y_train=numpy.array(data[:100,3])
x_test=numpy.array(data[:100,2])
y_test=numpy.array(data[:100,3])


xm,ym=x_train.mean(),y_train.mean()
b1_num,b1_denom=0,0
for i in range(len(x_train)):
    b1_num +=(x_train[i]-xm)*(y_train[i]-ym)
    b1_denom +=(x_train[i]-xm)**2
b1=b1_num/b1_denom 
b0=ym-b1*xm  

y_pred=b1*x_test + b0
plt.scatter(x_train,y_train,color='red',label='Training')
plt.scatter(x_test,y_pred,color='blue',label='Predict')
plt.scatter(x_test,y_pred,color='green',label='Actual values')
plt.legend()
plt.show()


sst,ssr=0,0
for i in range(len(y_test)):
    sst=(y.test[i]  . y_testmean())**2
    ssr=(y_test[i] . y_pred())**2
r=1-(float(ssr)/float(sst))
print('score ofLinear model :',r)