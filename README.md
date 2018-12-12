
# 使用超像素+紧密度算子+传统分类器的图像显著区域检测

环境: `Python 2.7`

依赖: `pip install boxx hpelm`;`conda install opencv`

测试: `python test.py`


Update in 2018.12

调用命令:
```bash
cd ../salience-ELM
source activate /home/yanglei/miniconda2
rlaunch -P 50 --cpu=1 --memory=15240  -- python algorithm.py
source deactivate
```



Update in 2018.08

 * 增加了 [Detecting Salient Objects via Color and Texture Compactness Hypotheses](https://ieeexplore.ieee.org/abstract/document/7523421/) 的原始 Saliency 方法.

注意: 

 1. `from boxx import *` 可能会和早期版本冲突
 2. `algorithm.buildMethodDic` 是个函数字典, 存储各个方法, 比较重要
 3. `coarseMethods=[]` 则为 Compactness 原文的原始方法

-------

# Old Version (2016)


## 基于ELM的图像显著区域检测
----
> 作者：`小磊`

> 邮箱：`ylxx@live.com`

> 时间：`2016-11-20`



## 环境

Python版本：`Python 2.7` with `Ipython`

常见库：`numpy`,`skiamge`

ELM库：
`hpelm`
> 下载地址:[`https://pypi.python.org/pypi/hpelm`](https://pypi.python.org/pypi/hpelm)（建议手动安装）

> 文档：[`http://hpelm.readthedocs.io/en/latest/api/elm.html`](http://hpelm.readthedocs.io/en/latest/api/elm.html)

数据集：[`http://202.118.75.4/lu/DUT-OMRON/index.htm`](http://202.118.75.4/lu/DUT-OMRON/index.htm)


## 注意事项

将数据集的`DUT-OMRON-image`和`pixelwiseGT-new-PNG`文件夹放到
项目父文件夹下

(参考调用代码`imgDir = '../DUT-OMRON-image/'`)
