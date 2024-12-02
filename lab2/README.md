**高级计算机系统架构：GPU异构计算实践**

**作业 1：maca-nbody kernel优化**
截止日期：2024年11月26日，星期三 - 上午10:00

**提交：**
用开发者账号登录沐曦算子竞赛平台(https://operator.metax-tech.com/)，上传修改后的项目工程，参加竞赛排名。

=================================================================================
**题目描述：maca-nbody kernel基础版实现方案**
=================================================================================
# nbody介绍

1. https://wikimili.com/en/N-body_problem 
2. https://baike.baidu.com/item/n%E4%BD%93%E9%97%AE%E9%A2%98/12713416

# maca-nbody基础代码

分布式并行计算实验代码，实验在沐曦的GPU上进行，利用maca对nbody算法进行优化。

网上搜索nbody的介绍，当前版本给出了最基础的并行（代码中也有一定的注释说明）。


## 编译和运行

```
mkdir build
cd build
cmake ..
make
./program
```
## 优化

修改[kernel.h]和[kernel.cpp]，部分可能的优化思路如下
1. 基础代码已经并行化，每个线程处理一个位置的body（参考[kernel.cpp]）。
2. 调整 BLOCK_SIZE得到相对最佳的值（测试后，思考为什么最佳）。
3. 自己控制内存的copy，申请和释放，消除缺页异常等。（mcMallocManaged -> mcMalloc / mcMallocHost）
4. 使用 shared_memory 进行优化，一个线程块共用一块 shared_memory，每个线程取部分数据提高数据访存效率。
5. 仔细观察 body_force 函数，会发现主要处理的最后为加法，可以尝试将原本的每个线程处理一个 body 进行改进，不同块中的多个线程共同处理一个 body 的数据信息，进一步提升并行率。
6. 可进一步尝试使用 shuffle 特性，看能否提升性能。

## References
1. https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
2. https://blog.csdn.net/EMF2423/article/details/121203018

