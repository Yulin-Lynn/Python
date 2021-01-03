"""
实验目的：
    使用 K 近邻算法识别手写数字
    URL：https://www.lanqiao.cn/courses/777/learning/?id=2621
"""

"""
数据描述：
    digits 目录下有两个文件夹，分别是:
    trainingDigits：训练数据，1934 个文件，每个数字大约 200 个文件。
    testDigits：测试数据，946 个文件，每个数字大约 100 个文件。
    每个文件中存储一个手写的数字，文件的命名类似 0_7.txt，
    第一个数字 0 表示文件中的手写数字是 0，后面的 7 是个序号。
"""
import numpy as np
import operator
# 用于返回指定的文件夹包含的文件或文件夹的名字的列表
from os import listdir 

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 下载数据集
# 在 Jupyter Notebook 单元格中执行，下载并解压数据。
!wget -nc "http://labfile.oss.aliyuncs.com/courses/777/digits.zip"
# 解压缩
!unzip -o digits.zip

# 把32x32个字符文本图像处理为1x1024向量
def img2vector(filename):
    # 创建1x1024向量
    returnVect = np.zeros((1,1024))

    fr = open(filename)
    for i in range(32):
        # 读取每一行
        lineStr = fr.readline()
        # 将每行前 32 字符转成 int 存入向量
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    
    return returnVect

img2vector('digits/testDigits/0_1.txt') 




"""
分类函数的步骤：
    计算欧氏距离
    当我们有一定的样本数据和这些数据所属的分类后，输入一个测试数据，
    就可以根据算法得出该测试数据属于哪个类别，此处的类别为 0-9 十个数字，就是十个类别。
    1.计算已知类别数据集中的点与当前点之间的距离；
    2.按照距离递增次序排序；
    3.选取与当前点距离最小的 k 个点；
    4.确定前 k 个点所在类别的出现频率；
    5.返回前 k 个点出现频率最高的类别作为当前点的预测分类。
"""
def classify0(inX, dataSet, labels, k):
    
    """
    参数: 
    - inX: 用于分类的输入向量
    - dataSet: 输入的训练样本集
    - labels: 样本数据的类标签向量
    - k: 用于选择最近邻居的数目
    """
    
    # 获取样本数据数量
    dataSetSize = dataSet.shape[0]

    # 矩阵运算，计算测试数据与每个样本数据对应数据项的差值
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    # sqDistances 上一步骤结果平方和
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)

    # 取平方根，得到距离向量
    distances = sqDistances**0.5

    # 按照距离从低到高排序
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 依次取出最近的样本数据
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 对类别出现的频次进行排序，从高到低
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回出现频次最高的类别
    return sortedClassCount[0][0]


group, labels = createDataSet()
classify0([0, 0], group, labels, 3)


"""
解题步骤:
    1.读取训练数据到向量（手写图片数据），从数据文件名中提取类别标签列表（每个向量对应的真实的数字）
    2.读取测试数据到向量，从数据文件名中提取类别标签
    3.执行K 近邻算法对测试数据进行测试，得到分类结果
    4.与实际的类别标签进行对比，记录分类错误率
    5.打印每个数据文件的分类数据及错误率作为最终的结果
"""
def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []

    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # 初始化样本数据矩阵（M*1024）
    trainingMat = np.zeros((m, 1024))

    # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        # 将样本数据存入矩阵
        trainingMat[i, :] = img2vector(
            'digits/trainingDigits/%s' % fileNameStr)

    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')

    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)

    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 提取数据向量
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)

        # 对数据文件进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        # 打印 K 近邻算法分类结果和真实的分类
        print("测试样本 %d, 分类器预测: %d, 真实类别: %d" %
              (i+1, classifierResult, classNumStr))

        # 判断K 近邻算法结果是否准确
        if (classifierResult != classNumStr):
            errorCount += 1.0

    # 打印错误率
    print("\n错误分类计数: %d" % errorCount)
    print("\n错误分类比例: %f" % (errorCount/float(mTest)))

handwritingClassTest()