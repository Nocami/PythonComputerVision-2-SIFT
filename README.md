# * 图像局部特征描述子
一幅图像是由固定个数等像素点组成等，每幅图像中总存在着其独特等像素点，我们称这些点为图像点特征点。今天要介绍点计算机视觉领域点图像特征匹配就是以各个图像的特征点为基础而进行的。本文介绍了两种特征点以及特征匹配方法。本文主要讲解局部图像描述子，即介绍用于图像匹配对两种描述子算法，阐述其原理，并且分别使用SIFT以及Harris对两幅图像检测匹配举例、通过SIFT匹配地理标记图像等实际操作。  
## 一.图像匹配  
图像匹配，就是通过找出所给出的每幅图像中点特征点，来与另外一幅图片中的特征点进行一一匹配，从而找到两幅或多幅图片之间的联系。常用的匹配方法有SIFT以及Harris两种，因SIFT(尺度不变特征变换)是过去十年中最成功的图像局部描述算子之一，故下文着重介绍SIFT。  
## 二.Harris角点  
Harris算子是一种角点特征，所谓角点，就是局部窗口沿各方向移动，均产生明显变化的点、图像局部曲率突变的点，典型的角点检测算法有：  
              • Harris角点检测  
              • CSS角点检测    
下图所示为“角点”：  
![image](https://github.com/Nocami/SIFT/blob/master/images/jiaodian.png)  
### 1.如何检测出Harris角点？  
角点检测最早期的想法就是取某个像素的一个邻域窗口。当这个窗口在像素点各个方向上进行移动时，观察窗口内平均的像素灰度值的变化，若变化巨大，则为角点，若单一方向无变化则为平滑，垂直方向变化大则为边缘。从下图可知，我们可以将一幅图像大致分为三个区域（‘flat’，‘edge’，‘corner’），这三个区域变化是不一样的。  
![image](https://github.com/Nocami/SIFT/blob/master/images/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-03-17%20%E4%B8%8B%E5%8D%886.40.21.png)  
其数学表达式为：  
将图像窗口平移[u,v]产生灰度变化E(u,v)  
![image](https://github.com/Nocami/SIFT/blob/master/images/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-03-17%20%E4%B8%8B%E5%8D%886.42.41.png)   
我们把图像域中点x上的对称半正定矩阵定义为：  
![image](https://github.com/Nocami/SIFT/blob/master/images/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-03-17%20%E4%B8%8B%E5%8D%886.45.42.png)  
M为自相关函数E(x,y)的近似Hessian矩阵(M为2*2矩阵)。  
设 λ1、λ2 λ1、λ2为M的特征值，定义角点相应函数R为：  
R=λ1λ2−k(λ1+λ2)2,即   
R=det(M)−k(tr(M))2  
det(M)=λ1*λ2  
tr(M)=λ1+λ2

### 2.图片匹配实例  
Harris.py代码如下：
```
#-*- coding: utf-8 -*-
from pylab import *
from PIL import Image

from PCV.localdescriptors import harris
from PCV.tools.imtools import imresize

"""
This is the Harris point matching example in Figure 2-2.
"""

#Figure 2-2上面的图
#im1 = array(Image.open("../data/crans_1_small.jpg").convert("L"))
#im2 = array(Image.open("../data/crans_2_small.jpg").convert("L"))

#Figure 2-2下面的图
im1 = array(Image.open("../data/sf_view1.jpg").convert("L"))
im2 = array(Image.open("../data/sf_view2.jpg").convert("L"))

#resize to make matching faster
im1 = imresize(im1, (im1.shape[1]/2, im1.shape[0]/2))
im2 = imresize(im2, (im2.shape[1]/2, im2.shape[0]/2))

wid = 5
harrisim = harris.compute_harris_response(im1, 5)
filtered_coords1 = harris.get_harris_points(harrisim, wid+1)
d1 = harris.get_descriptors(im1, filtered_coords1, wid)

harrisim = harris.compute_harris_response(im2, 5)
filtered_coords2 = harris.get_harris_points(harrisim, wid+1)
d2 = harris.get_descriptors(im2, filtered_coords2, wid)

print 'starting matching'
matches = harris.match_twosided(d1, d2)

figure()
gray() 
harris.plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
show()
```  
实例截图：  
![image](https://github.com/Nocami/SIFT/blob/master/images/Harris-02-04.jpg)  
![image](https://github.com/Nocami/SIFT/blob/master/images/Harris-10-11.jpg)  
![image](https://github.com/Nocami/SIFT/blob/master/images/Harris-y02-y03.jpg)  
## 三.SIFT(尺度不变特征变换)  
### 1.介绍：  
David Lowe在文献中提出的SIFT(尺度不变特征变换)是过去十年中最成功的图像局部描述子之一。SIFT经受住了时间的考验。SIFT特征包括兴趣点检测器和描述子，其描述子具有非常强点稳健型，这在很大程度上也是其能够成功和流行点原因。
![image](http://i0.qhmsg.com/dr/200__/t01fe92d0a98cf7c342.jpg)  
照片：David Lowe  
[SIFT解决的问题：](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#13-待办事宜-todo-列表)  

- [x] 目标的旋转、缩放、平移（rst）
- [x] 图像仿射/投影变换(视点viewpoint)
- [x] 弱光照影响(illumination)
- [x] 部分目标遮挡(occlusion)
- [x] 杂物场景(clutter)
- [x] 噪声  
### 2.SIFT算法的特点：  
1).SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性；
2). 区分性（Distinctiveness）好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配；
3). 多量性，即使少数的几个物体也可以产生大量的SIFT特征向量；
4).高速性，经优化的SIFT匹配算法甚至可以达到实时的要求；
5).可扩展性，可以很方便的与其他形式的特征向量进行联合。  

### 3.原理  
#### (一).尺度空间：  
在人体的视觉中，无论物体的大小，肉眼可以分辨出其相对大小。但是要让计算机掌握此能力却很困难。在没有标定的场景中，计算机视觉并不能计算出物体的大小，其中的一种方法是把物体不同尺度下的图像都提供给机器，让机器对物体在不同的尺度下有一个统一的认知。在建立统一认知的过程中，要考虑的就是在图像在不同的尺度下都存在的特征点。  
图-多分辨率图像金字塔：  
![image](http://www.opencv.org.cn/opencvdoc/2.3.2/html/_images/Pyramids_Tutorial_Pyramid_Theory.png)  
想象金字塔为一层一层的图像，层级越高，图像越小；尺度越大图像越模糊。
### 4.SIFT算法步骤：  
1).尺度空间极值检测：
搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。
2). 关键点定位
在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。
3). 方向确定
基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。
4). 关键点描述
在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。
