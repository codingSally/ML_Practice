import numpy as np

class SVD(object):

    def __init__(self):
        pass


    # 加载数据集
    def loadExData(self):
        return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
    def loadExData2(self):
        return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

    # 欧式距离计算相似度
    def ecludSim(inA,inB):
        return 1.0/(1.0 + np.la.norm(inA - inB))

    # 皮尔逊相关系数计算相似度
    def pearsSim(inA,inB):
        if len(inA) < 3 : return 1.0
        return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

    # 余弦定理计算相似度
    def cosSim(inA,inB):
        num = float(inA.T*inB)
        denom = np.la.norm(inA)*np.la.norm(inB)
        return 0.5+0.5*(num/denom)

    # 基于物品相似度的推荐引擎
    # 1. 计算在给定相似度计算方法的条件下，用户对物品的估计评分值
    def standEst(dataMat, user, simMeas, item):
        n = np.shape(dataMat)[1]
        simTotal = 0.0;
        ratSimTotal = 0.0
        for j in range(n):
            userRating = dataMat[user, j]
            if userRating == 0: continue
            overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, \
                                                dataMat[:, j].A > 0))[0]
            if len(overLap) == 0:
                similarity = 0
            else:
                similarity = simMeas(dataMat[overLap, item], \
                                     dataMat[overLap, j])
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0:
            return 0
        else:
            return ratSimTotal / simTotal

    # 2. 推荐引擎
    def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
        unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
        if len(unratedItems) == 0: return 'you rated everything'
        itemScores = []
        for item in unratedItems:
            estimatedScore = estMethod(dataMat, user, simMeas, item)
            itemScores.append((item, estimatedScore))
        return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

    # 基于CVD的评分估计
    def svdEst(dataMat, user, simMeas, item):
        n = np.shape(dataMat)[1]
        simTotal = 0.0;
        ratSimTotal = 0.0
        U, Sigma, VT = np.la.svd(dataMat)
        Sig4 = np.mat(np.eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix
        xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items
        for j in range(n):
            userRating = dataMat[user, j]
            if userRating == 0 or j == item: continue
            similarity = simMeas(xformedItems[item, :].T, \
                                 xformedItems[j, :].T)
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0:
            return 0
        else:
            return ratSimTotal / simTotal

    # 基于SVD的图像压缩
    def printMat(inMat, thresh=0.8):
        for i in range(32):
            for k in range(32):
                if float(inMat[i, k]) > thresh:
                    print
                    1,
                else:
                    print
                0,
            print
            ''

    def imgCompress(self, numSV=3, thresh=0.8):
        myl = []
        for line in open('0_5.txt').readlines():
            newRow = []
            for i in range(32):
                newRow.append(int(line[i]))
            myl.append(newRow)
        myMat = np.mat(myl)
        print
        "****original matrix******"
        self.printMat(myMat, thresh)
        U, Sigma, VT = np.la.svd(myMat)
        SigRecon = np.mat(np.zeros((numSV, numSV)))
        for k in range(numSV):  # construct diagonal matrix from vector
            SigRecon[k, k] = Sigma[k]
        reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
        print
        "****reconstructed matrix using %d singular values******" % numSV
        self.printMat(reconMat, thresh)