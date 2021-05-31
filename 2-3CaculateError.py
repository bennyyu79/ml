#程序2-3名称：CaculateError.py
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
Xi=np.array([162,165,159,173,157,175,161,164,172,158]) #身高数据
Yi=np.array([48,64,53,66,52,68,50,52,64,49])#体重数据
xy_res=[]
#计算残差
def residual(x,y):
    res=y-(0.42116973935*x-8.28830260655)
    return res
#读取残差
for d in range(0,len(Xi)):
    res=residual(Xi[d],Yi[d])
    xy_res.append(res)
##print(xy_res)
#计算残差平方和:越小表明拟合情况越好
xy_res_sum=np.dot(xy_res,xy_res)
#print(xy_res_sum) 
#如果数据拟合模型效果好，残差应该遵从正态分布(0,d*d:这里d表示残差)
#绘制样本点
fig=plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
ax=fig.add_subplot(111)
fig=qqplot(np.array(xy_res),line='q',ax=ax)
plt.show()

