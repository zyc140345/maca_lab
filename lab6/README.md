**高级计算机系统结构：GPU异构计算实践**

**作业 6：SGEMM kernel优化**
截止日期：2024年12月10日，星期二 - 上午10:00

**提交：**
用开发者账号登录沐曦算子竞赛平台(https://operator.metax-tech.com/)，上传修改后的项目工程，参加竞赛排名。

=================================================================================
**题目描述：SGEMM kernel优化**
=================================================================================
# SGEMM介绍

SGEMM（Single-precision General Matrix-Matrix multiplication）是一种基于BLAS（Basic Linear Algebra Subprograms）库的优化算法，专门用于执行单精度浮点数的矩阵乘法运算。SGEMM在多种场景中都能大放异彩，尤其是在以下领域：
- 大规模机器学习模型训练：在训练深度学习模型时，SGEMM可以高效地处理大量的矩阵运算。
- 高性能科学计算：在科学计算中，SGEMM可以用于解决线性方程组、计算特征值和特征向量等。
- 图像处理：在图像处理领域，SGEMM可以用于图像滤波、图像变换等操作。
- 任何依赖于密集矩阵运算的应用：SGEMM可以显著加速这些应用中的矩阵乘法运算。

# SGEMM kernel基础代码

[kernel.h]和[kernel.cpp]给出了最基础的SGEMM kernel代码，根据课程中学习的知识，网上搜索SGEMM kernel优化的相关文章，对SGEMM kernel进行优化。


## 编译和运行

```
mkdir build
cd build
cmake ..
make
./program
```
## 优化

修改[kernel.h]和[kernel.cpp]，尝试优化SGEMM kernel。SGEMM的性能优化涉及多个方面，包括但不限于：
- 内存访问对齐（GMEM Coalescing）：确保内存访问的连续性，提高内存访问效率。
- 块状划分（Blocktiling）：通过将矩阵划分为小块，可以减少缓存未命中，提高计算效率。
- Bank冲突避免：优化共享内存访问，避免Bank冲突，提高内存访问速度。
- 双缓冲技术：通过使用双缓冲技术，可以隐藏内存访问延迟，提高计算效率。
- 向量化内存访问：通过向量化内存访问，可以进一步提高内存访问效率。
- 自适应调优（Autotuning）：根据具体的硬件和矩阵尺寸，自动选择最优的配置和算法。

