import numpy as np

class PCA(object):

    def __init__(self):
        pass

    # 定义数据集
    def loadDataSet(fileName, delim='\t'):
        fr = open(fileName)
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [map(float,line) for line in stringArr]
        return np.mat(datArr)

    # PCA算法实现
    def pca(dataMat, topNfeat=9999999):
        meanVals = np.mean(dataMat, axis=0)
        # 去除平均值
        meanRemoved = dataMat - meanVals
        # 计算协方差矩阵
        covMat = np.cov(meanRemoved, rowvar=0)
        # 计算协方差矩阵的特征值和特征向量
        eigVals,eigVects = np.linalg.eig(np.mat(covMat))
        # 将特征值从大到小排序
        eigValInd = np.argsort(eigVals)
        # 保留前面个的N个特征值
        eigValInd = eigValInd[:-(topNfeat+1):-1]
        redEigVects = eigVects[:,eigValInd]
        # 将数据转换到上述N个特征向量构建的新空间中
        lowDDataMat = meanRemoved * redEigVects
        reconMat = (lowDDataMat * redEigVects.T) + meanVals
        return lowDDataMat, reconMat

    # 将NaN替换为平均值
    def replaceNanWithMean(self):
        datMat = self.loadDataSet('secom.data', ' ')
        numFeat = np.shape(datMat)[1]
        for i in range(numFeat):
            meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
            datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
        return datMat
