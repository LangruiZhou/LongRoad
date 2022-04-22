# L1-Week2 神经网络基础

**学习内容**：1、一些DeepLearning的常用技巧；2、为何DeepLearning有正向传播和反向传播？

以Logistic Regression为例进行阐述，该回归算法常用于解决二分分类（Binary Classification）问题

## 一、二分分类问题引入：判断“猫图”

1、机内图片如何表示：**RGB像素矩阵**

<div align="center"><img src="../../TyporaPics/1-16506374002911.png" alt="1" style="zoom:67%;" /></div>

2、将RGB矩阵转为用于输入的**特征向量x**：逐个矩阵、逐行地将元素值抄入一个**列向量**中

<div align="center"><img src="../../TyporaPics/2-16506374566582.png" alt="2" style="zoom:67%;" /></div>



## 二、一些常用术语及缩写

1、$n_x$：特征向量x的维度

2、$(x,y)$：表示一个样本，其中x为输入向量，y为输出向量

3、$\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$：表示一个有m个样本的数据集

4、 $m_{train}$：训练集的样本数量。 $m_{test}$：测试集的样本数量

5、样本矩阵：一般使用列向量来存储样本会比较方便。此时，样本集矩阵X的行数为样本向量的维度 $n_x$，列数为样本集中的样本数量 m。

    若想通过python得知样本矩阵X的规模，则可调用命令：`X.shape`，其返回结果为$(n_x,m)$

<div align="center"><img src="../../TyporaPics/3-16506374979913.png" alt="3" style="zoom:67%;" /></div>

## 三、逻辑回归 （Logistic Regression）

### 1、Logistic Regression概述

Logistic Regression被用于解决“二分类问题”。以猫图问题为例：给定一个$n$维向量 $x\in R^{n_x}$（它代表一张图片），我们希望得知这张图片是一张“猫图”的概率，即求 $\hat{y}=P(y=1|x)\in [0,1]$ 。这就是Logistic Regression的任务。

### 2、逻辑回归的假设函数（Hypothesis Function）

Logistic Regression的Hypothesis Function一般由线性函数与Sigmoid函数复合而成：

- **线性函数：$z=\omega^Tx+b$**

> $\omega\in R^{n_x}$是特征权重（weights），$b\in R$是偏差值
> 
- **Sigmoid函数**：$\sigma(z)=\frac{1}{1+e^{-z}}$，其图像如下所示

> Sigmoid函数可以修正线性函数，将其改为非线性函数，并使输出的预测值在[0,1]的范围内（表示一个概率）
> 

<div align="center"><img src="../../TyporaPics/4.png" alt="4" style="zoom: 50%;" /></div>

**综上所述**：$\hat{y}=\sigma(\omega^T x+b)$，而Logistic Regression的任务就是不断地学习参数 $\omega$ 和 $b$ ，以提高 $\hat{y}$ 的准确率。

