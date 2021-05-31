#程序2-1名称：scatterdiagram.py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
##样本数据(Xi,Yi)，需要转换成数组(列表)形式
Xi=np.array([162,165,159,173,157,175,161,164,172,158]) #身高数据
Yi=np.array([48,64,53,66,52,68,50,52,64,49])#体重数据
#画样本点
plt.figure(figsize=(8,6)) ##指定图像比例为8：6
plt.scatter(Xi,Yi,color="green",label="身高体重样本数据：",linewidth=1)
plt.xlabel('Height:cm')
plt.ylabel('Weight:kg')
plt.show()
