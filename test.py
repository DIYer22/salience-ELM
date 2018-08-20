# -*- coding: utf-8 -*-
from __future__ import unicode_literals
'''
函数功能测试
'''
from algorithm import io,np,sk,da
from algorithm import show,random,listdir,mark_boundaries,loga
from algorithm import getSlic,readImg
import algorithm as alg



a=random(3,5)
G = {} # G is a global var to save value for DEBUG in funcation
def setModuleConstant(module):
    module.IMG_DIR=IMG_DIR
    module.COARSE_DIR=COARSE_DIR
    module.IMG_NAME_LIST=IMG_NAME_LIST
    
#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'

IMG_DIR =  '../DataSet1/Imgs/'
COARSE_DIR ='../DataSet1/Saliency/'   
   
IMG_DIR =  r'G:\Data\HKU-IS/Imgs/'
COARSE_DIR =r'G:\Data\HKU-IS/Saliency/'


IMG_DIR = r'C:\D\dataset\saliency\ECSSD-small\images/'
COARSE_DIR =r'C:\D\dataset\saliency\ECSSD-small\ground_truth_mas/'

IMG_DIR = r'test/'
COARSE_DIR ='test/'


IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))

setModuleConstant(alg)

imgName = IMG_NAME_LIST[0]
img,imgGt = readImg(imgName)
rgbImg = img

rgbImg = np.zeros((100,100,3))
rgbImg[25:75,25:75,1:]=1.
#show(rgbImg)

def integratImgsBy3wayTest():
    from algorithm import getSlic,getCoarseDic,integratImgsBy3way,buildMethodDic
    labelMap = getSlic(rgbImg,300)
    img = rgbImg
    m, n = labelMap.shape 
    
    coarseMethods = ['GT']
    coarseDic = getCoarseDic(imgName,coarseMethods)
#    show(coarseDic)   
    coarseImgs=coarseDic.values()       
    
    g()
    refinedImgs=map(lambda f:f(img,coarseImgs,labelMap),buildMethodDic.values())
    img = integratImgsBy3way(refinedImgs)
    show(img)

def my4test():
#if 1:
    from algorithm import *
    labelMap = getSlic(rgbImg,200)
    maxLabel = labelMap.max()+1
    m, n = labelMap.shape 
    
    coarseMethods = ['MEAN']
    coarseDic = getCoarseDic(imgName,coarseMethods)
#    show(coarseDic)   
    sumCoarseImg = getSumCoarseImg(coarseDic)
    coarseImgs=coarseDic.values()    

    img = sk.color.rgb2lab(rgbImg)
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    diffEdges,diffNeighbors = getAllDiffEdgeAndNeighbor(labelMap,Ws)
    vectors = np.append(weightSumVectors,diffEdges,1)
    vectors = np.append(vectors,diffNeighbors,1)

    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    show(mark_boundaries(sk.color.lab2rgb(img),labelMap))
    print diffEdges.shape
    show(valueToLabelMap(labelMap,diffEdges.sum(1)))
    show(valueToLabelMap(labelMap,diffNeighbors[:,:4].sum(1)))
#    show(valueToLabelMap(labelMap,diffNeighbors[:,4:].sum(1)))
#    show(valueToLabelMap(labelMap,diffNeighbors.sum(1)))
    show(valueToLabelMap(labelMap,vectors[:,4:8].sum(1)))
#my4test()

def getEdgeNeighborTest():
    from algorithm import getEdge,getNeighborMatrix,getNeighbor
    
    labelMap = getSlic(img,200) 
    show(mark_boundaries(img,labelMap),0)
    edge = getEdge(labelMap)
    neighborMatrix = getNeighborMatrix(labelMap)    
    m, n = labelMap.shape 
    dic = getNeighbor(0,labelMap,neighborMatrix,4)
    print dic
    imgg = np.zeros((m,n))
    for k in dic:
        imgg[labelMap==k]=dic[k]
    show(imgg)

    imgg = np.zeros((m,n))
    for k in edge:
        imgg[labelMap==k]=1
    show(imgg)
    edge = getEdge(labelMap,0.06)
    imgg = np.zeros((m,n))
    for k in edge:
        imgg[labelMap==k]=1
    print 'edge width 0.06'
    show(imgg)    

#getEdgeNeighborTest()
def getDistanceTest():
    from algorithm import getDistance
    
    m,n=20,10
    ma = np.zeros((m,n)).astype(int)
    ma[:5,5:]=1
    ma[5:,:5]=2
    ma[5:,5:]=3
    
    dis = getDistance(ma)
    print ma
    print dis[0][2]
    print dis[0][1]


def getWeightSumTest():
    from algorithm import getLbp ,getVectors,getWeightSum
    labelMap = getSlic(img,200) 
    maxLabel = labelMap.max() + 1
    im = sk.color.rgb2lab(img)
    degreeVectors, Ws = getVectors(im, labelMap)
    vectors = getWeightSum(labelMap, degreeVectors, Ws)
    m,n = labelMap.shape
    imgg = np.zeros((m,n))
    imgg2 = np.zeros((m,n))
    order = ['lab','l','a','b','lab-texture','l-texture','a-texture','b-texture']
    labs = [im]+ [im[:,:,i] for i in range(3)]
    lbps = map(lambda c: getLbp(c,labelMap,1)[1],labs)
    labLbp = labs + lbps
    for color in range(vectors.shape[1]):
        for k in range(maxLabel):
            imgg[labelMap==k]=vectors[k][color]
            imgg2[labelMap==k]=degreeVectors[k][color]
        print order[color],'raw | scatter | weight sum'
#        show(sk.exposure.equalize_hist(imgg))
        show([labLbp[color],imgg2,imgg],1)
    loga(degreeVectors)
    loga(vectors)
    g()





def getRefindImgsTest():
    IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
    COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'
      
    IMG_DIR =  'test/'
    COARSE_DIR ='test/'
  
    IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
    coarseMethods = ['QCUT','DRFI']
    coarseMethods = ['GT']
    imgInd = 0
    n_segments,compactness = 200,10
    
    imgName = IMG_NAME_LIST[imgInd]
    img,imgGt = readImg(imgName)
    coarseDic = getCoarseDic(imgName,coarseMethods)
    #show(coarsesDic)   
    sumCoarseImg = getSumCoarseImg(coarseDic)
    coarseImgs=coarseDic.values()
    labelMap = getSlic(img,n_segments,compactness)
    
    rgb = img
    
    img = sk.color.rgb2lab(img)
    #show([mark_boundaries(img,labelMap),imgGt])
    # 获得4+4维  distance
    degreeVectors, Ws = getVectors(img, labelMap)
    vectors = getWeightSum(labelMap, degreeVectors, Ws)
    
    vectorsTrains = []
    coarseTrains = []
    for coarseImg in coarseImgs:
        coarseTrain, vectorsTrainTag = getCoarseTrain(coarseImg, labelMap)
        vectorsTrains += list(vectors[vectorsTrainTag])
        coarseTrains += list(coarseTrain)
    
    elm = getElm(np.array(vectorsTrains), np.array(coarseTrains))
    refined = elm.predict(vectors)[:,0]
    refinedImg = valueToLabelMap(labelMap,normalizing(refined))
    
    vectorsImg = valueToLabelMap(labelMap,normalizing(vectors.sum(1)))
    show([rgb,refinedImg])
    show([rgb,vectorsImg])
    show(vectorsImg-refinedImg)
    loga(vectorsImg-refinedImg)
    g()
def grabCutTest():
    coarseMethods = ['MY4','QCUT','DRFI']
    imgInd = 1
    n_segments,compactness = 200,10
    imgName = IMG_NAME_LIST[imgInd]
    img,imgGt = readImg(imgName)
    coarseDic = getCoarseDic(imgName,coarseMethods)
    refinedImg = coarseDic['DRFI']
    
    mask = grabCut(img,refinedImg)
    imgCut = img*mask[:,:,np.newaxis]
    
    show([img,imgCut])
    show([refinedImg,mask])
    
    
if __name__ == "__main__":
    from algorithm import *
    
#    getWeightSumTest()
#    getRefindImgsTest()

    coarseMethods=[] # 为空 则为 Compactness 原始方法
    
    buildMethods=['BG1']
    buildMethods=['MY1']
#    num = len(IMG_NAME_LIST)
#    num = 0
    for name in IMG_NAME_LIST[::]:
        buildImgs(name,buildMethods,coarseMethods)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
