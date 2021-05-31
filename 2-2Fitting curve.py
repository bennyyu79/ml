#程序2-2名称：Fittingcurve.py
import numpy as np
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

##样本数据(Xi,Yi)，需要转换成数组(列表)形式
Xi=np.array([162,165,159,173,157,175,161,164,172,158]) #身高数据
Yi=np.array([48,64,53,66,52,68,50,52,64,49])#体重数据
##需要拟合的函数func
def func(p,x):
    k,b=p
    return k*x+b
##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p,x,y):
    return func(p,x)-y
#k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
p0=[1,20]
#把error函数中除了p0以外的参数打包到args中
Para=leastsq(error,p0,args=(Xi,Yi))
#读取结果
k,b=Para[0]
print("k=",k,"b=",b)
#画样本点
plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
plt.scatter(Xi,Yi,color="red",label="Sample data",linewidth=2) 
#画拟合直线
x=np.linspace(150,180,80) #在150-180直接画80个连续点
y=k*x+b ##函数式
plt.plot(x,y,color="blue",label="Fitting Curve",linewidth=2) 
plt.legend() #绘制图例
plt.xlabel('横轴：身高（厘米）', fontproperties = 'simHei', fontsize = 12)
plt.ylabel('纵轴：体重（公斤）', fontproperties = 'simHei', fontsize = 12)
plt.show()


