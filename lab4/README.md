**高级计算机系统架构：GPU异构计算实践**

**作业 3：image-mean-filtering kernel优化**
截止日期：2024年12月3日，星期三 - 上午10:00

**提交：**
用开发者账号登录沐曦算子竞赛平台(https://operator.metax-tech.com/)，上传修改后的项目工程，参加竞赛排名。

=================================================================================
**题目描述：image-mean-filtering kernel优化**
=================================================================================
# image-mean-filtering介绍
图像平滑滤波(均值滤波、中值滤波和高斯滤波等)是到图像增强最常用的手段之一。
(https://blog.csdn.net/zaishuiyifangxym/article/details/89788020)

我们以图像均值滤波为例，它通过计算像素邻域内的平均值来平滑图像，可以有效地减少高频噪声，同时在某些情况下保留边缘信息。均值滤波会导致图像细节的模糊，因此在需要保留图像细节的情况下不适用。
(https://blog.csdn.net/qq_50380073/article/details/131436757）

均值滤波算法介绍：https://blog.csdn.net/weixin_51571728/article/details/121455266

# image-mean-filtering基础代码

图像均值滤波代码(image_image_mean_filtering_kernel)基础版实现参看kernel.cpp文件。
## 编译和运行

```
mkdir build
cd build
cmake ..
make
./program
```
## 优化

根据课堂知识、自学、网上查阅资料等学习途径尝试进行优化。


