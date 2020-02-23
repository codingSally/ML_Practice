from math import log

class Trees(object):
    
    # 计算香农熵 -- 开始处理数据集时，首先要测量集合中数据的不一致性，即熵
    def calcShannonEnt(dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob, 2)
            
        return shannonEnt
    
    
    # 创建数据集
    def createDataSet():
        dataSet = [[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
        
        labels = ['no surfacing', 'flippers']
        return dataSet, labels
    
    
    # 划分数据集
    # dataSet待划分的数据集、axis 划分数据集的特征 、value特征的返回值
    def splitDataSet(dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
                
        return retDataSet
    
    
    # 选择最好的数据集划分方式
    def chooseBestFeatureToSplit(self, dataSet):
        # 1. 计算原始香农熵
        numFeatures = len(dataSet[0]) - 1 
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        # 2. 划分数据集，计算熵
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            # 信息增益    
            infoGain = baseEntropy - newEntropy
            
            # 3. 选择信息增益最大的
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
                
        return bestFeature
    
    # 投票选举
    def majorityCnt(classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():classCount[vote] = 0
            classCount[vote] += 1
            
        sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse = True)
        
        return sortedClassCount[0][0]
    
    
    # 创建决策树
    def createTree(self, dataSet,labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList): 
            return classList[0]
        if len(dataSet[0]) == 1: 
            return majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(self, dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]      
            myTree[bestFeatLabel][value] = self.createTree(self, self.splitDataSet(dataSet, bestFeat, value),subLabels)
        return myTree   
            
     
    # 分类
    def classify(self, inputTree,featLabels,testVec):
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(self, secondDict[key], featLabels, testVec)
                else: classLabel = secondDict[key]
                    
        return classLabel


    # 序列化决策树 -- 决策树复用
    def storeTree(inputTree,filename):
        import pickle
        # py3中需要写成wb,不然报错
        fw = open(filename,'wb')
        pickle.dump(inputTree,fw)
        fw.close()
    
    def grabTree(filename):
        import pickle
        # py3中需要写成rb,不然报错
        fr = open(filename,'rb')
        return pickle.load(fr)
    
    
    
    
    
    
    
    
    
            