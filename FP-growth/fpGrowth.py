import numpy as np

class FP_Growth(object):

    def __init__(self):
        pass

    # 定义树节点
    class treeNode:
        def __init__(self, nameValue, numOccur, parentNode):
            self.name = nameValue
            self.count = numOccur
            self.nodeLink = None
            self.parent = parentNode      #needs to be updated
            self.children = {}
    
        def inc(self, numOccur):
            self.count += numOccur
        
        def disp(self, ind=1):
            print('  '*ind, self.name, ' ', self.count)
            for child in self.children.values():
                child.disp(ind+1)

    # 创建树
    def createTree(self,dataSet, minSup=1):
        headerTable = {}
        #go over dataSet twice
        for trans in dataSet:#first pass counts frequency of occurance
            for item in trans:
                headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
        for k in headerTable.keys():  #remove items not meeting minSup
            if headerTable[k] < minSup:
                del(headerTable[k])
        freqItemSet = set(headerTable.keys())
        #print 'freqItemSet: ',freqItemSet
        if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
        for k in headerTable:
            headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link
        #print 'headerTable: ',headerTable
        retTree = self.treeNode('Null Set', 1, None) #create tree
        for tranSet, count in dataSet.items():  #go through dataset 2nd time
            localD = {}
            for item in tranSet:  #put transaction items in order
                if item in freqItemSet:
                    localD[item] = headerTable[item][0]
            if len(localD) > 0:
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
                self.updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
        return retTree, headerTable #return tree and header table

    # 更新树
    def updateTree(self, items, inTree, headerTable, count):
        if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
            inTree.children[items[0]].inc(count) #incrament count
        else:   #add items[0] to inTree.children
            inTree.children[items[0]] = self.treeNode(items[0], count, inTree)
            if headerTable[items[0]][1] == None: #update header table
                headerTable[items[0]][1] = inTree.children[items[0]]
            else:
                self.updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
        if len(items) > 1:#call updateTree() with remaining ordered items
            self.updateTree(items[1::], inTree.children[items[0]], headerTable, count)

    # 更新头节点
    def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
        while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
            nodeToTest = nodeToTest.nodeLink
        nodeToTest.nodeLink = targetNode

    # 回溯
    def ascendTree(self, leafNode, prefixPath): #ascends from leaf node to root
        if leafNode.parent != None:
            prefixPath.append(leafNode.name)
            self.ascendTree(leafNode.parent, prefixPath)

    # 遍历列表
    def findPrefixPath(self, basePat, treeNode): #treeNode comes from header table
        condPats = {}
        while treeNode != None:
            prefixPath = []
            self.ascendTree(treeNode, prefixPath)
            if len(prefixPath) > 1:
                condPats[frozenset(prefixPath[1:])] = treeNode.count
            treeNode = treeNode.nodeLink
        return condPats

    # 递归查找频繁项集
    def mineTree(self, inTree, headerTable, minSup, preFix, freqItemList):
        bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
        for basePat in bigL:  #start from bottom of header table
            newFreqSet = preFix.copy()
            newFreqSet.add(basePat)
            #print 'finalFrequent Item: ',newFreqSet    #append to set
            freqItemList.append(newFreqSet)
            condPattBases = self.findPrefixPath(basePat, headerTable[basePat][1])
            #print 'condPattBases :',basePat, condPattBases
            #2. construct cond FP-tree from cond. pattern base
            myCondTree, myHead = self.createTree(condPattBases, minSup)
            #print 'head from conditional tree: ', myHead
            if myHead != None: #3. mine cond. FP-tree
                #print 'conditional tree for: ',newFreqSet
                #myCondTree.disp(1)
                self.mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

    def loadSimpDat(self):
        simpDat = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return simpDat

    def createInitSet(dataSet):
        retDict = {}
        for trans in dataSet:
            retDict[frozenset(trans)] = 1
        return retDict
