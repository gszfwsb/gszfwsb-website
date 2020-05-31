---
title: 深度学习pytorch(fastai) note1-五分钟搭建一个宠物识别模型
subtitle: pytorch and fastai
date: 2020-05-31T09:53:05.493Z
summary: 第一篇学习笔记
draft: false
featured: true
authors:
  - Shaobo Wang
tags:
  - pytorch(fastai)实战
categories:
  - pytorch(fastai)实战
image:
  filename: 5.jpg
  focal_point: ""
  preview_only: false
---
<!--StartFragment-->

## 说在前面

准备工作：

1. 3天采购硬件+自学atx装机（说一下显卡吧，GTX-1660 supper 显存６G. 小伙伴最好买一个８G+的）操作系统为Ubuntu 18.04 能不用windows就不用windows.
2. 一段时间的fastai-course-v3(2019)学习.\
   3.一年左右的machine learning 和 deep learning学习（以后来一个资源汇总）\
   4.jupyter notebook：史上最佳"ide"，即将会看到如果在jupyter notebook中打造一个自己的python库。（顺便一提，最近出了一款deepnote可能效果会比notebook更好一些，不过还是用熟悉的notebook来写这个系列好了。）

## 开始

先用一些魔法函数保证我们导入库和一些基本的配置。以后每次写一个新的项目都需要在jupyter notebook中这样弄

```ipynb
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

## 获取数据

```python
from fastai import *
from fastai.vision import *
```

## 下载数据

```ipynb
path = untar_data(URLs.PETS);
path
PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet')
path.ls()
[PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/annotations'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/crappy'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images')]
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
fnames[:5]
[PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images/Russian_Blue_142.jpg'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images/Persian_162.jpg'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images/Bombay_118.jpg'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images/pomeranian_153.jpg'),
 PosixPath('/home/shaobowang/.fastai/data/oxford-iiit-pet/images/yorkshire_terrier_186.jpg')]
```

## 构造图片数据集

1. 利用regex构造取出文件（正则表达式）
2. 利用ImageDataBunch

```python
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg'
```

这一段regex就是要取出来 /字母串_数字.jpg 这种样式的名称 [^/]是去掉/的意思

```python
data = ImageDataBunch.from_name_re(path_img,fnames,pat,ds_tfms=get_transforms(),size=224)
data.normalize(imagenet_stats)
ImageDataBunch;

Train: LabelList (5912 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
Russian_Blue,Persian,Bombay,shiba_inu,wheaten_terrier
Path: /home/shaobowang/.fastai/data/oxford-iiit-pet/images;

Valid: LabelList (1478 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
beagle,keeshond,miniature_pinscher,miniature_pinscher,great_pyrenees
Path: /home/shaobowang/.fastai/data/oxford-iiit-pet/images;

Test: None
```

ImageDataBunch是一个高内聚的API,fastai这种DataBunch类型的API还有很多

目前DL的一大技术缺陷就是对于图像，必须弄成一个特定的大小224,因为224 = 7*2^5,\
最后一层输出是7*7. 所以希望大小是这个样子

同时，我们对于图像进行一些随机的翻转或者镜像等，相当于做一个data augmentation

```python
data.show_batch(rows=3,figsize=(7,6))
```

![瞅一眼数据集](https://pic4.zhimg.com/80/v2-1573dfd84c541d65691c4e574c0ad807_1440w.jpg "瞅一眼数据集")

瞅一眼数据集

```python
print(data.classes)
['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
len(data.classes),data.c
(37, 37)
```

data.c的意思实际上是最后一层之前矩阵的一个输入，由于最后一层输出是７＊７所以实际上最后一层之前的weight是(N,7)这种状态的(矩阵乘法基本规则)，fastai中data.c就是这样一种设计。

## 训练

1. fastai中训练API主要是learner
2. create_cnn 是构造一个cnn网络
3. learn = cnn_learner(data, base_arch, metrics)

* data 是提供的数据 databunch
* base_arch是模型的架构, fastai中可以直接导入预训练的模型作为接口，比如resnet，一般我们都会使用resnet“试水”，而且往往效果都不错 - metrics有很多可以用的,比如 error_rate, accuracy等, 用来打印

```python
learn = cnn_learner(data,models.resnet34, metrics = error_rate)
```

## 拟合

借助于transfer learning 我们可以开始学习了, 我们与训练的模型已经在ImageNet上面“看”过了

```python
learn.fit_one_cycle(5)
```

![first train](https://pic2.zhimg.com/80/v2-878919e4b077f37e453a62e6666a2e6d_1440w.jpg)

第一次训练

我们使用的方法叫做fit_one_cycle()，这个比一般的fit效果更好，有一个super convergence的效果，具体的我们之后会提到，这里只需要知道它的效果比fit要来的快速强大，**可以看到，经过四轮之后我们的模型准确率已经达到了93%左右**

值得一体的是，我们现在还处于欠拟合的状态，因为train_loss > valid_loss，不过我们先不管了，我们随手写了些代码就已经如此优秀。

## 保存模型（权重）

使用learn.save()我们可以保存一下当前训练完成之后模型的参数，以便于下一次再复用

使用learn.load()我们可以加载之前保存的参数

```python
learn.save('stage-1')
```

## 分析结果（可视化）

1. 分析一下最容易混淆的图片，对他们进行一个排序和错误正确结果的对比
2. 分析一下混淆矩阵

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(15,11))
```

![](https://pic2.zhimg.com/80/v2-e66708c6b0bab5fe4d6311f3ea47e1f5_1440w.jpg)

最大误差的几个数据点

```python3
interp.most_confused(min_val=2)
```

结果：

```ipynb
[('Egyptian_Mau', 'Bengal', 5),
 ('Ragdoll', 'Birman', 5),
 ('Russian_Blue', 'British_Shorthair', 5),
 ('staffordshire_bull_terrier', 'american_pit_bull_terrier', 5),
 ('Siamese', 'Birman', 4),
 ('american_pit_bull_terrier', 'staffordshire_bull_terrier', 4),
 ('miniature_pinscher', 'chihuahua', 4),
 ('Bengal', 'Egyptian_Mau', 3),
 ('British_Shorthair', 'Russian_Blue', 3),
 ('american_bulldog', 'american_pit_bull_terrier', 3),
 ('english_setter', 'english_cocker_spaniel', 3),
 ('havanese', 'scottish_terrier', 3),
 ('Birman', 'Siamese', 2),
 ('Maine_Coon', 'Ragdoll', 2),
 ('Persian', 'British_Shorthair', 2),
 ('Ragdoll', 'Siamese', 2),
 ('boxer', 'american_bulldog', 2),
 ('havanese', 'newfoundland', 2),
 ('newfoundland', 'english_cocker_spaniel', 2)]
```

找到两两之间最容易混淆的

```python
interp.plot_confusion_matrix(figsize=(8,8))
```

![](https://pic3.zhimg.com/80/v2-148e199c831f73879e71cf6eb2d819ca_1440w.jpg)

混淆矩阵

## Fine tune模型和learning rate

跑完模型之后，learn就会“冻结”，需要手动激活然后就可以继续跑了

```python
learn.unfreeze()
learn.fit_one_cycle(1)
```

![](https://pic4.zhimg.com/80/v2-52cabb88ab84107e7e80ae2c35ce09f3_1440w.jpg)

效果变差了？

为什么这个效果比较差呢？因为我们解冻了整个模型，而实际上deep learning中的first layer往往只能学到一个很浅薄的东西，好像你第一次变成写出来的不过hello world而已，所以想要让第一层效果提升很多着实困难，因此，有显著效果的层往往是最后面的几层。我们训练整个模型的时候，实际上就是用相同的效果施加于整个模型（虽然learning rate有变化我们姑且理想地这样认为）

> 效果变差了，我们可以回去!

```python3
learn.load('stage-1')
Learner(data=ImageDataBunch;

Train: LabelList (5912 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
Russian_Blue,Persian,Bombay,shiba_inu,wheaten_terrier
Path: /home/shaobowang/.fastai/data/oxford-iiit-pet/images;

Valid: LabelList (1478 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
beagle,keeshond,miniature_pinscher,miniature_pinscher,great_pyrenees
Path: /home/shaobowang/.fastai/data/oxford-iiit-pet/images;

Test: None, model=Sequential(
......  
)
```

## 学习率探测器——又来一项黑科技

learn.lr_find()可以找出来学习率

learn.recorder.plot()可以画出来学习率的曲线

```python3
learn.lr_find()
learn.recorder.plot()
```

![](https://pic1.zhimg.com/80/v2-664e884494fd28b033bee7e8e0e83284_1440w.jpg)

lr_find()

LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.

![](https://pic1.zhimg.com/80/v2-3a684de1891936af7a37b2e4af92fc64_1440w.jpg)

recorder寻找学习率

我们画出来曲线之后，可以看到，第一个不在下降的点大概是8e-4，默认学习率是1e-3因此学习率有点大了，我们需要把学习率弄小一点。但是我们又希望最后几层学习率大一点，因为对一开始只会写hello world的阶段你努力太多没啥用，应该期末考试前再猛学一把 （狗头）

选择学习率的技巧如下：\

1. mar_lr=slice(..) 如果有两个参数，第一个是开始下降的点的至少1/10，比如这个8e-4，我们选的要比8e-5小，我们选5e-5 ；第二个参数选第一次用的1e-3(default)再除以10就是1e-4\
2. 如果只有一 个参数表示下限是省略的\
3. slice API:\
   slice(start=None,stop,step=None)

* start (optional) -Starting integer where the slicing of the object starts. Default to None if not provided.
* stop -Integer until which the slicing takes place. The slicing stops at index stop -1 (last element).
* step (optional) -Integer value which determines the increment between each index for slicing. Defaults to None if not provided.

![](https://pic2.zhimg.com/80/v2-390514e4015f0de0bededa436b19dd85_1440w.jpg)

继续训练！

yep 我们做到了94%+的准确率！（而且并没有怎么努力）

## 试试更强大的ResNet50

如果你的显存够，可以试试resnet50，可是我只有6G的显存。。。

> 第一节deep learning实战到此结束。如果本文对你有帮助，请不要吝啬赞同和喜欢。另外希望大家可以指出本文的错误以相互学习。谢谢！

<!--EndFragment-->