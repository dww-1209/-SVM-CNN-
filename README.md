# classify-2000photos-by-SVM-nad-CNN
本次实验采用的数据集大小为2000张图片，其中将50%作为训练集，50%作为测试集。训练集和测试集中的样本都有20个类别的图片，每个类别50张，总计1000张图片。

idx2label = {
    0:'African people and villages',
    1:'Beach',
    2:'Historical buildings',
    3:'Buses',
    4:'Dinosaurs',
    5:'Elephants',
    6:'Flowers',
    7:'Horses',
    8:'Mountains and glacier',
    9:'Food',
    10:'Dogs',
    11:'Lizards',
    12:'Fashion',
    13:'Sunsets',
    14:'Cars',
    15:'WaterFalls',
    16:'Antiques',
    17:'Battle ships',
    18:'Skiing',
    19:'Desserts'
}

## 本次实验通过Pytorch的torchvision.transforms来对数据进行数据增强，从而使得模型有更强的泛化能力。数据增强的方面分别包括：
随机裁剪大小
随机仿射变化
随机旋转角度
归一化

## 深度学习的方法
先通过深度学习的卷积神经网络中Resnet18作为迁移学习的目标主题，然后进行微调，得到模型准确率

## 传统机器学习
利用机器学习常用的SVM模型来作为我们的分类器，首先要做的就是特征提取

### 特征提取
1.	GLCM特征（见GLCM.py）
2.	HIST特征(见HIST.py)
3.	LBP特征(LBP.py)
4.	SIFT特征(SIFT.py)
5.	GLCM与HIST混合特征
6.	LBP与SIFT混合特征
7.	所有特征的混合特征

### 模型训练与降维

部分特征的维度可能过长，此时可以考虑采用pca技术进行降维，利用降维后的特征向量来训练模型

### 特征融合（可选）

融合部分特征作为新特征，重新训练SVM分类器，观察其表现

## 集成学习
将深度学习的CNN与机器学习所得到的众多SVM分类器集成，采用加权软投票的方式，得到一个强分类器

## 实验结论简单汇总

1.	集成学习强分类器：94%
2.	深度模型CNN ：  80.2%
3.	HIST特征	   ：  36.6%
4.	HIST和LBP混合特征：34%
5.	GLCM特征	　　：20.6%
6.	LBP特征		：18.7%
7.	SIFT特征		：10.5%
8.	所有特征混合	：10.5%
