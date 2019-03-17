# 图像局部特征描述子
一幅图像是由固定个数等像素点组成等，每幅图像中总存在着其独特等像素点，我们称这些点为图像点特征点。今天要介绍点计算机视觉领域点图像特征匹配就是以各个图像的特征点为基础而进行的。本文介绍了两种特征点以及特征匹配方法。本文主要讲解局部图像描述子，即介绍用于图像匹配对两种描述子算法，阐述其原理，并且分别使用SIFT以及Harris对两幅图像检测匹配举例、通过SIFT匹配地理标记图像等实际操作。  
## 图像匹配  
图像匹配，就是通过找出所给出的每幅图像中点特征点，来与另外一幅图片中的特征点进行一一匹配，从而找到两幅或多幅图片之间的联系。常用的匹配方法有SIFT以及Harris两种，因SIFT(尺度不变特征变换)是过去十年中最成功的图像局部描述算子之一，故下文着重介绍SIFT。  
## Harris角点  
Harris算子是一种角点特征，所谓角点，就是局部窗口沿各方向移动，均产生明显变化的点、图像局部曲率突变的点，典型的角点检测算法有：  
              • Harris角点检测  
              • CSS角点检测    
下图所示为“角点”：  
![image](https://img-blog.csdn.net/20141223222604093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGFuZGFuXzM5Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  
### 如何检测出Harris角点？  
角点检测最早期的想法就是取某个像素的一个邻域窗口。当这个窗口在像素点各个方向上进行移动时，观察窗口内平均的像素灰度值的变化，若变化巨大，则为角点，若单一方向无变化则为平滑，垂直方向变化大则为边缘。从下图可知，我们可以将一幅图像大致分为三个区域（‘flat’，‘edge’，‘corner’），这三个区域变化是不一样的。  
![image](https://img-blog.csdn.net/20141223222933456?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGFuZGFuXzM5Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  

