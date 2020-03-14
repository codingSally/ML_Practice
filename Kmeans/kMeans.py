import numpy as np

class kMeans(object):

    def __init__(self):
        pass


    # 加载数据集
    def loadDataSet(fileName):
        dataMat = []
        fr = open(fileName)
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine)
            dataMat.append(fltLine)
        return dataMat

    # 计算欧氏距离
    def distEclud(vecA, vecB):
        return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

    # 生成随机质心
    def randCent(self, dataSet, K):
        n = np.shape(dataSet)[1]
        centroids = np.mat(np.zeros((K, n)))
        for j in range(n):
            minJ = np.min(dataSet[:, j])
            rangeJ = float(np.max(dataSet[:, j]) - minJ)
            centroids[:, j] = minJ + rangeJ * np.random.rand(K, 1)
        return centroids

    # kMeans聚类算法
    def kMeans(dataSet, K, distMeas = distEclud, createCent = randCent):
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m, 2)))
        centroids = createCent(dataSet, K)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = inf, minIndex = -1
                for j in range(K):
                    distJI = distMeas(centroids[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True;
                clusterAssment[i, :] = minIndex, minDist ** 2

            print(centroids)
            for cent in range(K):
                # 获取这个簇下面的所有值
                ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
                # 计算平均值
                centroids[cent, :] = np.mean(ptsInClust, axis=0)

        return centroids, clusterAssment

    # 二分K-均值
    def binkMeans(self, dataSet, K, distMeas=distEclud):
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m, 2)))
        centroid0 = np.mean(dataSet, axis=0).tolist()[0]
        centList = [centroid0]
        for j in range(m):
            clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
        # 划分数据
        while(len(centList) < K):
            lowestSSE = np.inf
            for i in range(len(centList)):
                # 获取当前簇中所有点
                ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
                centroidMat, splitClustAss = self.kMeans(ptsInCurrCluster, 2, distMeas)

                sseSplit = np.sum(splitClustAss[:, 1])
                sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
                print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
                if(sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
            print('the bestCentToSplit is: ', bestCentToSplit)
            print('the len of bestClustAss is: ', len(bestClustAss))
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
            centList.append(bestNewCents[1, :].tolist()[0])
            clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
        return np.mat(centList), clusterAssment




