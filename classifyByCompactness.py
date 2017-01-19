# -*- coding: utf-8 -*-

from algorithm import *

def saveCompactness(imgName,
                    segmentList=[200,250,750]
                    ):
    compactness=20 
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    img,imgGt = readImg(imgName)
    rgb = img
    img = sk.color.rgb2lab(img)
    compactnessImgs = []
    for n_segments in segmentList:
        labelMap = getSlic(rgb,n_segments,compactness)  
        # 获得4+4维  distance
        degreeVectors, Ws = getVectors(img, labelMap)
        weightSumVectors = getWeightSum(labelMap, degreeVectors, Ws)
        compactness = valueToLabelMap(labelMap,weightSumVectors)
        compactnessImgs += [compactness]
#        show(compactness[...,:].sum(2))
    compactness = (sum(compactnessImgs))/3.
    show([mark_boundaries(rgb,labelMap),mark_boundaries(compactness[...,:].sum(2),imgGt==0)])
    saveData(compactness,LABEL_DATA_DIR+'%s.compactness'%imgName)

def getCompactnessVector(imgName):
    '''
    Divide the img's compactness into w*h equal regions
    return a w*h*8 dim vector
    '''
    w, h = 8, 8 # num of blocks  in row and col
    if isinstance(imgName, int):
        imgName = IMG_NAME_LIST[imgName]
    print 'img index:%d/%d'%(IMG_NAME_LIST.index(imgName),len(IMG_NAME_LIST))
    compactness = loadData(LABEL_DATA_DIR+'%s.compactness'%imgName)
    m, n = compactness.shape[:2] 
    maskM = np.array([[i]*n for i in range(m)])
    maskN = np.array([range(n) for i in range(m)])
    vector = []
    for i in range(h):
        for j in range(w):
            ind = np.where((i*m/h<=maskM)&(maskM<(i+1)*m/h)&(j*n/w<=maskN)&(maskN<(j+1)*n/w),True,False)
#            show(ind)            
            vector += list(np.mean(compactness[ind],0))
    vector = np.array(vector)
    return vector

def getCompactnessTrain(dataDic,aucMethod='ME1'):
    '''
    dataDic:getPrCurve 生成的含有auc详细信息的dic
    return train:list of [name,(1,0)] or  [name,(0,1)] to train
    '''
    data = dataDic['img']
    aucs = np.array([data[name][aucMethod]['auc'] for name in data])
    maxx,minn = aucs.max(),aucs.min()
    d = maxx - minn
    omega = np.mean(aucs) - minn
    alpha = 0.8
    '''有改动 把tl的min换成max'''
    th = min([0.9*d+minn,(1+alpha)*omega+minn])
    tl = max([0.6*d+minn,(1-alpha)*omega+minn])
#    loga(aucs)
#    print th,tl,maxx,minn
    train = []
    for name in data:
        auc = data[name][aucMethod]['auc']
        if auc > th:
            train += [[name,(1,0)]]
        if auc < tl:
            train += [[name,(0,1)]]
    return train
    
    
#for name in IMG_NAME_LIST:
#    saveCompactness(name)
if __name__ == '__main__':

    aucMethod = 'DRFI'
    dataDic = loadData('bad')
    train =getCompactnessTrain(dataDic,aucMethod='ME1')
    vectors = []
    ys = []
    for name,y in train:
        print name
        vector = getCompactnessVector(name)
        vectors += [vector]
        ys += [y]
        
    vectors,ys = np.array(vectors).astype(np.float64),np.array(ys)
    print vectors.shape,np.mean(vectors,0),ys.sum(0)
    elm = getElm(vectors,ys)
    rel = elm.predict(np.array([vector]))[:,0]








