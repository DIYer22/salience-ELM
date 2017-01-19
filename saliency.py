# -*- coding: utf-8 -*-
from os import listdir
import os
import multiprocessing
from copy import deepcopy

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt  
import skimage as sk
import skimage.io as io
from skimage import data as da
from skimage.segmentation import mark_boundaries
import cv2

import cProfile as profile
runt = profile.run

from tools import mapp,show,getPoltName,performance,normalizing,random
from tools import saveData,loadData

from warnings import filterwarnings
filterwarnings('ignore')



a=random(3,5)
G = {} # G is a global var to save value for DEBUG in funcation



#IMG_DIR =  '../DataSet1/Imgs/'
#COARSE_DIR ='../DataSet1/Saliency/'

#IMG_DIR =  r'G:\Data\HKU-IS/Imgs/'
#COARSE_DIR =r'G:\Data\HKU-IS/Saliency/'

#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet1\Saliency/'
    
#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DataSet2\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DataSet2\Saliency/'

IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\HKU-IS\Imgs/'
COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\HKU-IS\Saliency/'

#IMG_DIR = r'E:\3-experiment\SalBenchmark-master\Data\DUT-OMRON-image\Imgs/'
#COARSE_DIR =r'E:\3-experiment\SalBenchmark-master\Data\DUT-OMRON-image\Saliency/'


LABEL_DATA_DIR = os.path.dirname(IMG_DIR[:-1])+'/LabelData/'
if not os.path.isdir(LABEL_DATA_DIR):
    os.mkdir(LABEL_DATA_DIR)
    
IMG_NAME_LIST = filter(lambda x:x[-3:]=='jpg',listdir(IMG_DIR))
allMethods = list(set(map(lambda x: x[x.rindex('_')+1:-4],listdir(COARSE_DIR))))

if __name__ == '__main__':
    from algorithm import readImg,getCoarseDic,showpr
    import algorithm as alg  # main algorithm
    from algorithm import buildImgs,mergeImgs
    
    from analysis import plotMethods

    
    IMG_NAME_LIST=IMG_NAME_LIST[::]
    # build MY* 创造
    coarseMethods=['DISC2','QCUT']
    
    buildMethods=['MY4']
    num = len(IMG_NAME_LIST)
    num = 0
    for name in IMG_NAME_LIST[:num]:
        buildImgs(name,buildMethods,coarseMethods)
    
    
    # merge methods to make ME* 融合
    mergeMethods=['MY4','DISC2']
    for name in IMG_NAME_LIST[:num]:
        mergeImgs(name,mergeMethods)
        
    
    #  画图分析
    #raise LookupError,u'结束'
    #showMethods = ["MY1","MY2","MY3","MY4","MY5","ME1","ME2","ME3","MEAN", "DRFI", "GMR","QCUT","DISC2"]
    showMethods = ["MY4","ME1","FT","GC","RC", "DRFI", "GMR","QCUT","DISC2"]
    showMethods += filter(lambda x:x not in showMethods,buildMethods)
    
    data = plotMethods(IMG_DIR,COARSE_DIR,showMethods,
                       num=num,
                       save=getPoltName(coarseMethods,IMG_DIR)
                       )
#    dic = deepcopy(data['img'])
#    for name in dic:
#        dic[name] = {'method':dic[name]}
#        _dic = dic[name]
#        _dic['aucs'] = [(method, v['auc']) for method,v in _dic['method'].items()]
#        _dic['var'] = np.var([v['auc'] for method,v in _dic['method'].items()])
#    
#    varss = [(k,v['var'],v['aucs']) for k,v in dic.items()]
#    varss.sort(key=lambda x:x[1])
    
    #[showpr(i,showMethods) for i in range(num)]
    #aucs = data['MEAN']['imgs']
    #aucs = aucs.items()
    #aucs.sort(key=lambda x:x[1])
    #bad = filter(lambda x:x[1]<0.5,aucs)
    #saveData(bad)
    
    #bad = loadData('bad')        
    #for name,auc in bad[::-1]:
    #    print name    
    #    showpr(name)
    
    










