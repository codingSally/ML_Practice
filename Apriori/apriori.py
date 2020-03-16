import numpy as np

class Apriori(object):

    def __init__(self):
        pass

    # 创建测试集
    def loadDataSet(self):
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    # 创建候选项集
    def createC1(dataSet):
        C1 = []
        for transaction in dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
                
        C1.sort()
        # frozenset 不可变集合
        return map(frozenset, C1)

    # 扫描频繁项及支持度
    def scanD(D, Ck, minSupport):
        ssCnt = {}
        for tid in D:
            for can in Ck:
                if can.issubset(tid):
                    if not ssCnt.has_key(can): ssCnt[can]=1
                    else: ssCnt[can] += 1
        numItems = float(len(D))
        retList = []
        supportData = {}
        for key in ssCnt:
            support = ssCnt[key]/numItems
            if support >= minSupport:
                retList.insert(0,key)
            supportData[key] = support
        return retList, supportData

    # 创建候选项集
    def aprioriGen(Lk, k):
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):
            for j in range(i+1, lenLk):
                L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
                L1.sort(); L2.sort()
                if L1==L2:
                    retList.append(Lk[i] | Lk[j])
        return retList

    # 主函数
    def apriori(self, dataSet, minSupport = 0.5):
        C1 = self.createC1(dataSet)
        D = map(set, dataSet)
        L1, supportData = self.scanD(D, C1, minSupport)
        L = [L1]
        k = 2
        while (len(L[k-2]) > 0):
            Ck = self.aprioriGen(L[k-2], k)
            Lk, supK = self.scanD(D, Ck, minSupport)
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    # 关联规则生成函数
    def generateRules(self, L, supportData, minConf=0.7):
        bigRuleList = []
        for i in range(1, len(L)):
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
                else:
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList

    # 计算关联规则置信度
    def calcConf(freqSet, H, supportData, brl, minConf=0.7):
        prunedH = []
        for conseq in H:
            conf = supportData[freqSet]/supportData[freqSet-conseq]
            if conf >= minConf:
                print(freqSet-conseq,'-->',conseq,'conf:',conf)
                brl.append((freqSet-conseq, conseq, conf))
                prunedH.append(conseq)
        return prunedH

    # 生成更多的关联规则
    def rulesFromConseq(self, freqSet, H, supportData, brl, minConf=0.7):
        m = len(H[0])
        if (len(freqSet) > (m + 1)):
            Hmp1 = self.aprioriGen(H, m+1)
            Hmp1 = self.calcConf(freqSet, Hmp1, supportData, brl, minConf)
            if (len(Hmp1) > 1):
                self.rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

    # 打印规则
    def pntRules(ruleList, itemMeaning):
        for ruleTup in ruleList:
            for item in ruleTup[0]:
                print(itemMeaning[item])
                print("           -------->")
            for item in ruleTup[1]:
                print(itemMeaning[item])
            print("confidence: %f" % ruleTup[2])
            print

