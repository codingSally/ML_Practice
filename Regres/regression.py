import numpy as np

class Regress(object):

    def __init__(self):
        pass

    # 加载数据集
    def loadDataSet(fileName):
        numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
        dataMat = [];
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat


    # 标准回归函数
    def standRegres(self, xArr, yArr):
        xMat = np.mat(xArr); yMat = np.mat(yArr).T
        xTx = xMat.T * xMat
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T*yMat)
        return ws


    # 局部线性加权函数
    def lwlr(testPoint, xArr, yArr, k=1.0):
        xMat = np.mat(xArr);
        yMat = np.mat(yArr).T
        m = np.shape(xMat)[0]
        weights = np.mat(np.eye((m)))
        for j in range(m):
            diffMat = testPoint - xMat[j, :]
            # 这里为什么会乘以 deffMat.T?
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        if np.linalg.det(xTx) == 0.0:
            print
            "This matrix is singular, cannot do inverse"
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return testPoint * ws

    # 岭回归
    def ridgeRegres(xMat, yMat, lam=0.2):
        xTx = xMat.T * xMat
        denom = xTx + np.eye(np.shape(xMat)[1]) * lam
        if np.linalg.det(denom) == 0.0:
            print
            "This matrix is singular, cannot do inverse"
            return
        ws = denom.I * (xMat.T * yMat)
        return ws

    def ridgeTest(xArr, yArr):
        xMat = np.mat(xArr);
        yMat = np.mat(yArr).T
        yMean = np.mean(yMat, 0)
        yMat = yMat - yMean  # to eliminate X0 take mean off of Y
        # regularize X's
        xMeans = np.mean(xMat, 0)  # calc mean then subtract it off
        xVar = np.var(xMat, 0)  # calc variance of Xi then divide by it
        xMat = (xMat - xMeans) / xVar
        numTestPts = 30
        wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
        for i in range(numTestPts):
            ws = np.ridgeRegres(xMat, yMat, np.exp(i - 10))
            wMat[i, :] = ws.T
        return wMat


    # 标准化
    def regularize(xMat):  # regularize by columns
        inMat = xMat.copy()
        inMeans = np.mean(inMat, 0)  # calc mean then subtract it off
        inVar = np.var(inMat, 0)  # calc variance of Xi then divide by it
        inMat = (inMat - inMeans) / inVar
        return inMat

    # 前向逐步线性回归
    def stageWise(self, xArr, yArr, eps=0.01, numIt=100):
        xMat = np.mat(xArr);
        yMat = np.mat(yArr).T
        yMean = np.mean(yMat, 0)
        yMat = yMat - yMean  # can also regularize ys but will get smaller coef
        xMat = self.regularize(xMat)
        m, n = np.shape(xMat)
        returnMat = np.zeros((numIt, n))  # testing code remove
        ws = np.zeros((n, 1));
        wsTest = ws.copy();
        wsMax = ws.copy()
        for i in range(numIt):  # could change this to while loop
            # print ws.T
            lowestError = np.inf;
            for j in range(n):
                for sign in [-1, 1]:
                    wsTest = ws.copy()
                    wsTest[j] += eps * sign
                    yTest = xMat * wsTest
                    rssE = self.rssError(yMat.A, yTest.A)
                    if rssE < lowestError:
                        lowestError = rssE
                        wsMax = wsTest
            ws = wsMax.copy()
            returnMat[i, :] = ws.T
        return returnMat