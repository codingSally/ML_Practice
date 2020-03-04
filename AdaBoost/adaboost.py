import pandas as pd
import numpy as np

class AdaBoost(object):
    def __init__(self):
        pass

    # 导入数据集
    def loadSimpData(self):
        datMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return datMat, classLabels


    # 单层决策树生成函数
    # 通过比较阈值对数据进行分类
    def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
        retArray = np.ones((np.shape(dataMatrix)[0],1))
        if threshIneq == 'lt':
            retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:, dimen] > threshVal] = -1.0

        return retArray


    # 构建生成函数
    def buildStump(self, dataArr,classLabels,D):
        dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
        m,n = np.shape(dataMatrix)
        numSteps = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
        minError = np.inf # 正无穷
        for i in range(n):
            rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
            setpSize = (rangeMax - rangeMin)/numSteps
            for j in range(-1, int(numSteps) + 1):
                for inequal in ['lt', 'gt']:
                    threshVal = (rangeMin + float(j) * setpSize)
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)
                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predictedVals == labelMat] = 0
                    weightedError = D.T * errArr
                    # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                    if weightedError < minError:
                        minError = weightedError
                        bestClassEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClassEst

    # 基于单层决策树的AdaBoost训练过程
    def adaBoostTrainDS(self, dataArr,classLabels,numIt=40):
        weakClassArr = []
        m = np.shape(dataArr)[0]
        D = np.mat(np.ones(m,1)/m)
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(numIt):
            bestStump, error, classEst = self.buildStump(dataArr, classLabels, D)
            print("D:", D.T)
            alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            print("classEst:" , classEst.T)
            expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
            D = np.multiply(D, np.exp(expon))
            D = D/D.sum()
            aggClassEst += alpha * classEst
            print("aggClassEst:" , aggClassEst.T)
            aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
            errorRate = aggErrors.sum()/m
            print("total error: ", errorRate)
            if(errorRate == 0):
                break
        return weakClassArr

    # 按照上述方法训练完弱分类器后，分类时按照弱分类器+权重进行判断
    # AdaBoost 分类函数
    def adaClassify(self, datToClass,classifierArr):
        detaMatrix = np.mat(datToClass)
        m = np.shape(detaMatrix)[0]
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(len(classifierArr)):
            classEst = self.stumpClassify(self.dataMatrix, classifierArr[i]['dim'], \
                                     classifierArr[i]['thresh'], \
                                     classifierArr[i]['ineq'])
            aggClassEst += classifierArr[i]['alpha'] * classEst

            print(aggClassEst)
        return np.sign(aggClassEst)
