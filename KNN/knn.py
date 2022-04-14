import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN
# 利用 csv 包进行对训练集的读操作
with open("traindata.csv",encoding='utf-8-sig',mode="r") as f:

    data = list(csv.reader(f))

    # 删除data 列表的第一行
    data.pop(0)
    # print(data)

# 利用np.array把列表转化为数组
data = np.array(data)

# 拿到 data 数组中的第三列，也就是训练集的第三列数据并存储，用以 knn 训练
Y_tarin = data[:,-1]

# 将 data 数组中的第三列删除并存储起来
X_tarin = np.delete(data,2,axis=1)

# 将 Y_tarin 数组中的数据进行元素替换并升维
#   因为.fit 函数中 y 只能使用二进制来训练因此需要
#   cocorrespond = {'A':0,'B':1} 定义一个替换对象用于接下来的数组替换
#   [correspond[i] if i in correspond else i for i in Y_tarin]将数组中的元素进行替换
#   reshape((50,1))将替换后数组进行升维操作，变成二维数组（50行，1 列）
correspond = {'A':0,'B':1}
Y_tarin = np.array([correspond[i] if i in correspond else i for i in Y_tarin]).reshape((50,1))
# print(X_tarin)
# print(Y_tarin)

# K 近邻分类器，n_neighbors为所选用的近邻数，相当于K
knn = kNN(n_neighbors=3)

# 使用 X_tarin 作为训练数据和 Y_tarin 作为目标值来训练模型
knn.fit(X_tarin,Y_tarin)

# 利用 csv 包进行对测试集的读操作
with open("testdata.csv",encoding='utf-8-sig',mode="r") as f:

    data = list(csv.reader(f))

    # 删除data 列表的第一行
    data.pop(0)
    # print(data)

# 利用 KNN 算法进行预测
result = knn.predict(data)

# 将前面替换的数据再替换回去
correspond = {0:'A',1:'B'}
result = np.array([correspond[i] if i in correspond else i for i in result])

# 输出预测数据
print(result)