import numpy as np

class Logistic(object):
    
    def __init__():
        pass
        
    
    # 加载数据
    def loadDataSet():
        dataMat = []; labelMat = []
        fr = open('testSet.txt')
        for line in fr.readlines():
            lineArr = line.strip().split()
            # 为什么把w0设置成1
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat,labelMat

    # sigmod函数
    def sigmoid(inX):
        return 1.0/( 1+ np.exp(-inX))
    
    # 获取梯度系数
    def gradAscent(self, dataMatIn, classLabels):
        dataMatrix = np.mat(dataMatIn) # 转换为numpy矩阵数据类型
        labelMat = np.mat(classLabels).transpose() # 转置矩阵
        m,n = np.shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500
        weights = np.ones((n,1))
        
        for k in range(maxCycles):
            h = self.sigmoid(dataMatrix * weights)
            error = (labelMat - h)
            weights = weights + alpha * dataMatrix.transpose() * error
            
        return weights
    
    
    # 随机梯度上升算法    
    def stocGradAscent0(self, dataMatrix, classLabels):
        m,n = np.shape(dataMatrix)
        alpha = 0.01
        weights = np.ones(n)   #initialize to all ones
        for i in range(m):
            h = self.sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        return weights
    
    
    # 改进的随机梯度上升算法
    def stocGradAscent1(self, dataMatrix, classLabels, numIter=150):
        m,n = np.shape(dataMatrix)
        weights = np.ones(n)   #initialize to all ones
        for j in range(numIter):
#             dataIndex = range(m)
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4/(1.0+j+i)+0.0001    
                # 随机选取一个样本点，来更新梯度系数
                # random.uniform随机生成下一个实数，它在 [x, y] 范围内。
                randIndex = int(np.random.uniform(0,len(dataIndex)))
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                # 使用完后删除，无放回方式
                del(dataIndex[randIndex])
        return weights
            
    
    #画出最佳拟合曲线
    def plotBestFit(self, weights):
        import matplotlib.pyplot as plt
        dataMat,labelMat = self.loadDataSet()
        dataArr = np.array(dataMat)
        n = np.shape(dataArr)[0] 
        xcord1 = []; ycord1 = []
        xcord2 = []; ycord2 = []
        for i in range(n):
            if int(labelMat[i])== 1:
                xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
            else:
                xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = np.arange(-3.0, 3.0, 0.1)
        # 为什么使用0= W0X0 + W1X1 + W2X2？[难道是因为针对多分类的时候，找的是最佳拟合曲线，而不是直线，所以二分类，默认X0 = 1] 
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
        plt.xlabel('X1'); plt.ylabel('X2');
        plt.show()
        
        
     # 分类器
    # 就需要把测试集上的每个特征向量乘以最优化方法得来的回归系数，再将乘机求和，最后输入到sigmoid函数中，
    # 如果对应的函数值大于0.5，则为1，否则为0
    def classifyVector(inX, weights):
        prob = self.sigmoid(sum(inX * weights))
        if prob > 0.5: return 1.0
        else: return 0.0
        
        
    # 书中的测试示例
    def colicTest(self):
        frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
        trainingSet = []; trainingLabels = []
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainWeights = self.stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
        errorCount = 0; numTestVec = 0.0
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr =[]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(self.classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
                errorCount += 1
        errorRate = (float(errorCount)/numTestVec)
        print("the error rate of this test is: %f" % errorRate)
        return errorRate

    def multiTest(self):
        numTests = 10; errorSum=0.0
        for k in range(numTests):
            errorSum += self.colicTest(self)
        print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        
    
        