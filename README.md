# 基于ELM的图像显著区域检测
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
