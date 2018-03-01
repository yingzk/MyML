### Decision Tree

​						$G(x) = \sum_{c=1}^{c}[b(x)=c]·G_{c}(x)$

$G_{x}$ 完整的树

$b(x)$ 分支的标准

$G_c(x)$ 子树，在第c个分支

一个树看为由多棵树组成

​							**tree = (root, sub-tree)**




**Function Decision Tree(data $D = {(x_n,y_n)}^{N}_{n=1}$)**

if 终止条件:

​	return base hypothesis $g_t(x)$;

else:

1. learn 一个分支标准

2. 将D 划分为C部分， $D_c={(x_n,y_n):b(x_n)=c}$ 

   (通过分支标准b，对$x_n$ 进行分支得到分支$D_c$)

3. 建立树G的子树 $G_c \leftarrow DecisionTree(D_c)$ 

4. 返回$G(x)=\sum^c_{c=1}[b(x)=c]G_c(x)$  树G由多个子树$G_c$组成



### Classification and Regression Tree(C&RT)

**Function Decision Tree(data $D={(x_nmy_n)}_(n=1^N)$)**

if 终止条件:

​	return base hypothesis $g_t(x)$

else:

- split D to C parts $D_c={(x_n, y_n):b(x_n)=c}$ 	

  将D划分为C部分

#####2中简单划分方式:

- C = 2(binary tree) 二叉树
- $g_t(x) = E_(in)-optimal \  constant$    回传的$g_t(x)$是一个常数
  - binary/multiclass classification (0/1 error): majority of {$y_n$}
  - regression (squared error): average of $\{y_n\}$



将数据划分为2份后，希望两边的纯度(purity)越高越好

最后回传的常数好或者不好（即为纯度）

求它的$E_(in)$

$E_(in)$  越低越好

**regression error**

$impurity(D) = \frac{1}{N}\sum_{n=1}^{N}(y_n-\bar{y})^2$

with $\bar{y} = average\ of\ \{y_n\}$

**classification error**

$impurity(D) = \frac{1}{N}\sum_{n=1}^{N}[y_n\ne y^{*}]$

with $y^{*}=majority\ of\ \{y_n\}$

![1](C:\Users\yzk13\Desktop\MachineLearning\Decision Tree\1.png)



### Gini指数（求纯度）

$Gini(D)=1-\sum_{k=1}^{K}p_k^2$

$K$ 一共K类

$p_k$  被正确分类

所以最后得到的Gini指数是指不纯度



![2](C:\Users\yzk13\Desktop\MachineLearning\Decision Tree\2.png)

由于只分为2类(K=2),一类为$\mu$ 则另一类为$1-\mu$

带入式子直接求得



实际不会采用$E_{in}$为0的树，$E_{in}$为0导致过拟合(overfit), 所以我们需要一个稍微大点的$E_{in}$值，需要进行剪枝处理(Pruning)

​						最终: $argmin( all\ possible\ G)\ E_{in}+\lambda\Omega(G)$

 



| 特征选择方案 |    算法    |
| :----: | :------: |
|  信息增益  |   ID3    |
| 信息增益率  | **C4.5** |
| Gini指数 | **C&RT** |





### 信息熵

表示所有样本中出现的不确定性之和，根据熵的概念，熵越大，不确定性就越大，所以把事情搞清楚就需要更多的信息量

​							$Ent(D) = - \sum^{|y|}_{k=1}p_klog_2p_k$

### 信息增益

**信息增益 = 熵 - 条件熵**

在这里就是  **信息增益 = 类别信息熵 - 属性信息熵**

表示信息不确定性的减少的程度。一个属性的信息增益越大，表示用这个属性可以更改的减少划分后样本的不确定性

​						$Gain(D,a)=Ent(D)-\sum_{v=1}^{V}\frac{D^v}{D}Ent(D^v)$



























