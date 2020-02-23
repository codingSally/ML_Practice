# class 文件中也必须import依赖，不沿用控制层的
from numpy import * 
import operator

class KNN(object):
    
    def __init__(self,data):
        self.data = data
        
    # 创建数据集
    def createDataSet():
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels = ['A','A','B','B']
    
        return group,labels
    
    
    # 定义分类器
    def classify0(inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 复制元素并做差
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        
        sortedDistIndicies = distances.argsort()
        classCount={}  #字典 or 集合 这里是字典
        
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #初始值是0，有对应字典元素的话，就+1
            
        #d.items() 返回迭代器  operator.itemgetter(1)指定排序索引
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        
        return sortedClassCount[0][0]
    
    # 将文本转成矩阵
    def file2matrix(filename):
        fr = open(filename)
        arrayOLines = fr.readlines()
        numberOLines = len(arrayOLines)
        returnMat = zeros((numberOLines, 3))
        classLabelVector = []
        
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            
            index += 1
            
        return returnMat,classLabelVector
    
    # 归一化特征值
    def autoNorm(dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        
        normDataSet = dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))
        
        return normDataSet,ranges,minVals
    
    # 图片转向量
    def img2vector(filename):
        returnVect = zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect
        
        
        
        
        
        
        
        
        
        