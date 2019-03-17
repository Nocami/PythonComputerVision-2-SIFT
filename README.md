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
