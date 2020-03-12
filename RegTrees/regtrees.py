import numpy as np

class RegTrees(object):

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

    # 划分数据集
    def binSplitDataSet(dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] >= value)[0], :][0]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] < value)[0], :][0]

        return mat0, mat1

    # 生成叶节点。当chooseBestSplit()函数确定不再对数据进行切分时，将调用该函数来得到叶节点的模型
    # 在回归树种，该模型其实就是目标变量的均值
    def regLeaf(dataSet):  # returns the value used for each leaf
        return np.mean(dataSet[:,-1])

    # 误差估计函数 -- 方差的均方误差总方差
    def regErr(dataSet):
        return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

    # 最佳划分函数
    def chooseBestSplit(self, dataSet, leafType = regLeaf, errType = regErr, ops=(1,4)):
        # tols为容许误差的最小值，toln为切分的最小样本数
        tolS = ops[0]; tolN = ops[1]
        #数据集最后一行为目标值，通过set去重，若去重后数组长度为1，则表明当前数据集为一类数据，当作叶节点处理
        if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
            return None, leafType(dataSet)

        m,n = np.shape(dataSet)
        S = errType(dataSet)
        bestS = np.inf; bestIndex = 0; bestValue = 0;
        for featureIndex in range(n-1):
            for splitVal in set(dataSet[:, featureIndex]):
                mat0, mat1 = self.binSplitDataSet(dataSet, featureIndex, splitVal)
                # 如果划分后的数据个数小于最小样本数，则跳过该循环
                if(np.shape(mat0)[0] < tolN) or (np.shape(mat1) < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featureIndex
                    bestValue = splitVal
                    bestS = newS

        # 如果当前误差变化的不是很大，则不划分，将整个数据集当作叶子节点处理
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        # 如果划分后的子集个数很小则不划分，将整个数据集当作叶子节点处理
        if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[1] < tolN):
            return None, leafType(dataSet)

        return bestIndex, bestValue

    # 创建树
    def createTree(self, dataSet, leafType = regLeaf, errType = regErr, ops=(1, 4)):
        feat, val = self.chooseBestSplit(self, dataSet, leafType, errType, ops)
        if feat == None:
            return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree['left'] = self.createTree(lSet, leafType, errType, ops)
        retTree['right'] = self.createTree((rSet, leafType, errType, ops))

        return retTree


    # 接下来模型树
    # 模型树的叶节点生成函数
    # 将数据集格式化成目标变量Y 和 自变量X
    def linearSolve(dataSet):
        m,n = np.shape(dataSet)
        X = np.mat(np.ones(m,n)); Y = np.mat(np.ones(m,1))
        X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
        xTx = X.T * X
        if np.linalg.det(xTx) == 0.0:
            raise NameError('This matrix is singular, cannot do inverse,\n\
                    try increasing the second value of ops')
        ws = xTx.I * (X.T * Y)
        return ws, X, Y

    # 负责生成叶节点模型
    def modelLeaf(self, dataSet):
        ws, X, Y = self.linearSolve(dataSet)
        return ws

    # 计算误差
    def modelErr(self, dataSet):
        ws, X, Y = self.linearSolve(dataSet)
        yHat = X * ws
        return np.sum(np.power(Y - yHat, 2))

    # 接下来是树回归的预测
    # 判断是不是树
    def isTree(obj):
        return (type(obj).__name__ == 'dict')

    def regTreeEval(model, inDat):
        return float(model)

    def modelTreeEval(model, inDat):
        n = np.shape(inDat)[1]
        X = np.mat(np.ones((1,n+1)))
        X[:,1:n+1]=inDat
        return float(X*model)
    # 自顶向下遍历树，直到命中叶节点为止
    def treeForCast(self, tree, inData, modelEval = regTreeEval):
        if not self.isTree(tree):
            return self.modelEval(tree, inData)

        if inData[tree['spInd']] > tree('spVal'):
            if self.isTree(tree['left']):
                return self.treeForeCast(tree['left'], inData, modelEval)
            else:
                return modelEval(tree['left'], inData)
        else:
            if self.isTree(tree['right']):
                return self.treeForeCast(tree['right'], inData, modelEval)
            else:
                return modelEval(tree['right'], inData)
    # 以一组向量形式返回预测值
    def createForeCast(self, tree, testData, modelEval=regTreeEval):
        m = len(testData)
        yHat = np.mat(np.zeros((m, 1)))
        for i in range(m):
            yHat[i, 0] = self.treeForeCast(tree, np.mat(testData[i]), modelEval)
        return yHat