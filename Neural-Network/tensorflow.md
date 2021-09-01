## TensorFlow入门

### TensorFlow计算模型—计算图

所有tensorflow的程序都可以通过计算图的形式来表示，计算图中的每一个节点代表一个运算，每一条边代表计算之间的依赖关系。

```python
# 一个简单的计算图定义
import tensorflow as tf
a = tf.constant([1.0, 2.0], dtype = tf.float32, name = "a")
b = tf.constant([3.0, 4.0], dtype = tf.float32, name = "b")
res = a + b
```

tensorflow程序一般可以分为两个阶段，第一阶段需要定义计算图中的所有计算，第二阶段为执行计算。在tensorflow程序中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图。

```python
# 查看张量所属计算图
print(a.graph)
print(tf.get_default_graph())
print(a.graph is tf.get_default_graph())
```

tensorflow支持通过tf.Graph函数来生成新的计算图。不同计算图上的张量和计算不会共享。

```python
# 定义计算图g1
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(name = 'v', shape = [1], initializer = tf.zeros_initializer)

# 定义计算图g2
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(name = 'v', shape = [1], initializer = tf.ones_initializer)

# 计算图g1中读取变量v的值
with tf.Session(graph = g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse = True):
        print(sess.run(tf.get_variable("v")))

# 计算图g2中读取变量v的值
with tf.Session(graph = g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse = True):
        print(sess.run(tf.get_variable("v")))
```

### TensorFlow数据模型—张量

在tensorflow程序中，所有的数据都通过张量（tensor）的形式来表示。从功能的角度上看，张量可以被简单理解为多维数组。其中零阶张量表示标量（scalar），也就是一个数；一阶张量为向量（vector）；二阶张量为一个二维数组；n阶张量可以理解为一个n维数组。张量在tensorflow中的实现并不是直接采用数组的形式，它只是对tensorflow中运算结果的引用。在张量中并没有真正保存数字，而保存的是如何得到这些数字的计算过程。

```python
# 张量信息
import tensorflow as tf
a = tf.constant([1.0, 2.0], dtype = tf.float32, name = "a")
print(a)
'''
输出：
Tensor("a:0", shape=(2,), dtype=float32)
'''
```

从代码运行结果可以看出，一个张量中主要保存了三个属性：名字（name）、维度（shape）和类型（type）。名字为一个张量的唯一标识符，张量的命名可以通过"node:src_output"的形式来给出。其中node为节点的名称，src_output表示当前张量来自节点的第几个输出。张量的第二个属性是维度，围绕张量的维度tensorflow给出了很多有用的运算。

```python
# 张量维度操作
import tensorflow as tf
t1 = tf.constant([[[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [-4, -5, -6]]], dtype = tf.int32, name = "t1")
t2 = tf.constant([[[7, 8, 9], [10, 11, 12]], [[-7, -8, -9], [-10, -11, -12]]], dtype = tf.int32, name = "t2")
with tf.Session() as sess:
    print(sess.run(tf.concat([t1, t2], 0)))
    print(sess.run(tf.concat([t1, t2], 1)))
    print(sess.run(tf.concat([t1, t2], 2)))

'''
输出：
[[[  1   2   3]
  [  4   5   6]]

 [[ -1  -2  -3]
  [ -4  -5  -6]]

 [[  7   8   9]
  [ 10  11  12]]

 [[ -7  -8  -9]
  [-10 -11 -12]]]
  
 
 [[[  1   2   3]
  [  4   5   6]
  [  7   8   9]
  [ 10  11  12]]

 [[ -1  -2  -3]
  [ -4  -5  -6]
  [ -7  -8  -9]
  [-10 -11 -12]]]
  
  
  [[[  1   2   3   7   8   9]
  [  4   5   6  10  11  12]]

 [[ -1  -2  -3  -7  -8  -9]
  [ -4  -5  -6 -10 -11 -12]]]
'''
```

上述代码中t1和t2为两个张量，维度为（2,2,3）。使用concat函数连接两个三维数组，当axis=0时表示第一个维度上的拼接操作，得到一个维度为（4,2,3）的张量；当axis=1时表示第二个维度上的拼接操作，得到一个维度为（2,4,3）的张量；当axis=2时表示第三个维度上的拼接操作，得到一个维度为（2,2,6）的张量。

张量的第三个属性是类型，每一个张量会有一个唯一的类型。tensorflow会对参与运算的所有张量进行类型检查，当发现类型不匹配时会报错。tensorflow支持14种不同的类型，主要包括了实数（tf.float32、tf.float64）、整数（tf.uint8、tf.int8、tf.int16、tf.int32、tf.int64）、布尔型（tf.bool）和复数（tf.complex64、tf.complex128）。

### TensorFlow运行模型—会话

在tensorflow中会话（session）用来执行计算图中定义好的运算，其拥有并管理tensorflow程序运行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源，否则可能出现资源泄露的问题。tensorflow中使用会话的模式一般有两种，第一种模式需要明确调用会话生成函数和会话关闭函数，该模式代码流程如下

```python
# 创建一个会话
sess = tf.Session()
# 使用会话得到关心的运算结果
sess.run(...)
# 关闭会话释放资源
sess.close()
```

使用此模式时，在所有计算完成之后，需要明确调用Session.close函数来关闭会话并释放资源。然而，当程序因为异常而退出时，关闭会话的函数可能就不会被执行从而导致资源泄露。为了解决异常退出时资源释放的问题，tensorflow可以通过python的上下文管理器来使用会话，该模式代码流程如下

```python
# 通过python上下文管理器管理会话
with tf.Session() as sess:
    sess.run(...)
# 上下文退出时会话关闭和资源释放自动完成
```



## 神经网络基础

人工神经网络（Artificial Neural Network，简称ANN），是20世纪80年代以来人工智能领域兴起的研究热点。它从信息处理的角度对人脑神经元网络进行抽象建模，按不同的连接方式组成不同的网络，在工程与学术界也常简称为神经网络或类神经网络。神经网络是有史以来发明的最优美的编程范式之一，近年来，神经网络相关的研究工作不断深入，其在模式识别、智能机器人、自动控制、生物、医学、经济等领域已成功地解决了许多现代计算机难以解决的实际问题，表现出了良好的智能特性。

### 感知器

感知器在20世纪五、六十年代由科学家Frank Rosenblatt发明，其受到Warren McCulloch和Walter Pitts早期工作的影响。感知器是如何工作的呢？一个感知器接受几个二进制输入，$x_1,x_2,...$，并产生一个二进制输出：

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115174102.png)

示例中的感知器有三个输入，$x_1,x_2,x_3$。Rosenblatt提议一个简单的规则来计算输出。引入权重$w_1,w_2,w_3$，表示相应输入对于输出的重要性。感知器的输出（0或者1），由分配权重后的总和$\sum_jw_jx_j$小于或大于某些阈值决定。精确的代数形式如下
$$
\begin{equation} 
output= 
\begin{cases} 
0\quad if \ \sum_jw_jx_j\leq threshold \\  
1\quad if \ \sum_jw_jx_j>threshold      
\end{cases} 
\end{equation}
$$
将感知器的阈值用偏置$b=-threshold$来代替，感知器的规则可以重写为
$$
\begin{equation}
output=
\begin{cases}
0 \quad if \ wx+b\leq0 \\
1 \quad if \ wx+b>0
\end{cases}
\end{equation}
$$
如下代码实现了一个简单的感知器类

```python
# 感知器类实现
import numpy as np

class perceptron(object):
	def __init__(self, sizes):
		self.sizes = sizes
		self.weights = np.random.randn(sizes,)
		self.biase = np.random.randn()

	def activation(self, x):
		if x < 0:
			return 0
		else:
			return 1
	
	def output(self, x):
		y = np.dot(self.weights, x) + self.biase
		return self.activation(y)
    
```



感知器是一种权衡依据做出决策的方法。感知器被采用的另一种方式是计算基本的逻辑功能，即我们通常认为的运算基础（"与"、"或"、"与非"）。假设我们有个两个输入的感知器，每个输入的权重为-2，整体的偏置为3。如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115174201.png)

可以得到：输入00产生输出1，即$(-2)\times0+(-2)\times0+3=3>0$是正数；输入11产生输出0，即$(-2)\times1+(-2)\times1+3=-1<0$是负数；输入01或10时产生输出1。感知器实现了一个与非门。实际上，感知器网络能够计算任何逻辑功能，因为与非门是通用运算，我们可以在多个与非门之上构建出任何运算。例如，我们能用与非门构建一个电路，它把两个二进制数$x_1$和$x_2$相加。

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115174241.png)

我们可以设计学习算法，能够自动调整人工神经元的权重和偏置，这种调整可以响应外部的刺激，而不需要程序员的直接干预。这些学习算法使得我们能够以一种根本区别于传统逻辑门的方式使用人工神经元。有别于显示地设计与非或其他门，神经网络能简单地学会解决问题，这些问题有时候直接用传统的电路设计是很难解决的。

### S型神经元

和感知器类似，S型神经元有多个输入，$x_1,x_2,...$。但这些输入可以取0到1中的任何值，而不仅仅是0或1。例如，0.638...是一个S型神经元的有效输入。同样，S型神经元对每个输入有对应的权重，$w_1,w_2,...$，和一个总的偏置$b$，但是输出不是0或1，而是$\sigma(wx+b)$，这里$\sigma$被称为S型函数，定义为
$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$
一个具有输入$x_1,x_2,...$，权重$w_1,w_2,...$，和偏置$b$的S型神经元的输出是
$$
\frac{1}{1+e^{-\sum_jw_jx_j-b}}
$$
为了理解和感知器模型的相似性，假设 $z=wx+b$ 是一个很大的正数，那么$e^{-z}\approx0$ 而 $\sigma(z)\approx1$。即当 $z=wx+b$ 很大并且为正时，S型神经元的输出近似为1，正好和感知器一样。相反地，假设 $z=wx+b$ 为一个很大的负数，那么$e^{-z}\rightarrow\infty$ ，$\sigma(z)\approx0$。所以当 $z=wx+b$ 很大并且为负时，S型神经元的行为也非常近似一个感知器。只有在 $z=wx+b$ 取中间值时，S型神经元和感知器模型有比较大的偏离。

### 激活函数

激活函数的作用是给神经网络加入一些非线性因素，由于线性模型的表达能力不够，激活函数的加入可以使得神经网络更好地解决较为复杂的实际问题。神经网络中常用的激活函数主要有阶跃函数、sigmoid函数、tanh函数、ReLU函数、LReLU和PReLU函数、ELU函数。

#### 阶跃函数

阶跃函数定义如下
$$
\begin{equation}
step(x)=
\begin{cases}
1\quad if \ x\geq0 \\
0\quad if \ x<0
\end{cases}
\end{equation}
$$
其图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\step.png)

如果激活函数为阶跃函数，那么输出会依赖于 $wx+b$ 的正负。感知器的激活函数为阶跃函数。

#### sigmoid函数

sigmoid函数定义如下
$$
\begin{equation}
sigmoid(x)=\frac{1}{1+e^{-x}}
\end{equation}
$$
其图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\sigmoid.png)

sigmoid函数将实数压缩到0~1区间，输入大的负数输出趋近于0，输入大的整数输出趋近于1。sigmoid函数由于其强大的解释力，常被用来表示神经元的活跃程度：从不活跃（0）到假设上的最大活跃（1）。在实践中，sigmoid函数经历了从受欢迎到不受欢迎的阶段，如今很少再被使用。sigmoid激励函数有两个主要缺点：

- 容易饱和，产生梯度消失的现象。sigmoid神经元的一个很差属性是神经元的活跃度在0和1处饱和，它的梯度在这些地方接近于0。在反向传播更新中，梯度是连乘的形式，因此如果某处的梯度过小，就会很大程度上出现梯度消失，使得几乎没有信号经过这个神经元。此外，人们必须额外注意sigmoid神经元权值的初始化来避免饱和。例如，当初始权值过大时，几乎所有的神经元都会饱和以至于网络几乎不能学习。
- sigmoid函数的输出不是零均值的，导致后层的神经元输入是非零均值的信号，这会对梯度产生影响。假设后层神经元的输入都为正（$x>0,f(x)=wx+b$），那么对权重 $w$ 求局部梯度则都为正，这样在反向传播的过程中 $w$ 要么都往正方向更新，要么都往负方向更新，导致有一种捆绑的效果，导致收敛缓慢。如果是按照batch的方式去训练，每个batch可能得到不同的正负符号，相加后可以缓解这个问题。

#### tanh函数

tanh函数定义如下
$$
\begin{equation}
tanh(x)=\frac{1-e^{-2x}}{1+e^{-2x}}
\end{equation}
$$
其图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\tanh.png)

tanh函数的输出范围为-1~1，因此基本是零均值的，实际中tanh函数比sigmoid函数更常用，但依然存在梯度消失的问题。

#### ReLU函数

ReLU是最近几年非常流行的激活函数，定义如下：

$$
\begin{equation}
ReLU(x)=
\begin{cases}
x\quad if \ x\geq0 \\
0\quad if \ x<0
\end{cases}
\end{equation}
$$
其图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\relu.png)

ReLU激励函数的优点如下

- 因为其线性、非饱和的形式，ReLU在梯度下降上与tanh/sigmoid相比具有更快的收敛速度。
- 不会出现梯度消失的问题。
- ReLU会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。
- sigmoid和tanh涉及到了很多指数的操作，而ReLU可以更加简单的实现。

ReLU激励函数的缺点如下

- 输出不是零均值的。
- 神经元坏死现象。因为ReLU函数在负数部分的梯度为0，某些神经元可能永远不会被激活，导致相应参数永远不会被更新。产生该现象的两个原因：(1) 参数初始化问题；(2) learning rate太大导致在训练过程中参数更新太大。解决方法：采用Xavier初始化方法，以及避免将学习率设置太大或使用adagrad等自动调节学习率的优化算法。
- ReLU不会对数据做幅度压缩，因此数据的幅度会随着模型层数的增加不断扩张。

#### LReLU和PReLU函数

LReLU是Leakly ReLU的简称，定义如下
$$
\begin{equation}
LReLU(x)=
\begin{cases}
x\quad if \ x\geq0 \\
ax\quad if \ x < 0
\end{cases}
\end{equation}
$$
参数 $a$ 一般是一个比较小的正数，例如0.01。PReLU函数是LReLU函数的改进，其区别在于参数 $a$ 是固定还是可变。当$a=0.1$时，LReLU函数的图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\lrelu.png)

LReLU用来解决ReLU带来的神经元坏死的问题，有实验证明，其表现不一定比ReLU好。

#### ELU函数

ELU函数定义如下
$$
\begin{equation}
ELU(x)=
\begin{cases}
x\quad  if\ x\geq0 \\
a(e^{x}-1)\quad if \ x<0
\end{cases}
\end{equation}
$$
参数 $a$ 是一个较小的正数，当$a=0.5$时，ELU函数的图形如下所示

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\ELU.png)

ELU继承了ReLU的优点，且不会有神经元坏死的问题，输出的均值接近0。但是计算量大，其表现不一定比ReLU好。

如下代码实现了一个支持多种激活函数的简单神经元类

```python
# 支持多种激活函数的神经元类实现
import numpy as np

class neuron(object):
	def __init__(self, sizes, func, a = 0.01):
		self.sizes = sizes
		self.func = func
		self.a = a
		self.weights = np.random.randn(sizes,)
		self.biase = np.random.randn()

	def activation(self, x):
		if self.func == 'step':
			if x < 0:
				return 0
			else:
				return 1
		elif self.func == 'sigmoid':
			return 1.0 / (1.0 + np.exp(-x))
		elif self.func == 'tanh':
			return (1.0 - np.exp(-2 * x)) / (1.0 + np.exp(-2 * x))
		elif self.func == 'relu':
			if x < 0:
				return 0
			else:
				return x
		elif self.func == 'lrelu':
			if x < 0:
				return (self.a) * x
			else:
				return x
		elif self.func == 'elu':
			if x < 0:
				return (self.a) * (np.exp(x) - 1)
			else:
				return x
		else:
			raise ValueError('Unknown neuronal activation function!')

	def output(self, x):
		y = np.dot(self.weights, x) + self.biase
		return self.activation(y)
```

### 前向传播算法

下图为一个四层全连接神经网络。网络中最左边的为输入层，其中的神经元称为输入神经元。网络中最右边的为输出层，其中的神经元称为输出神经元。输入层和输出层之间的为隐藏层，图中网络有两个神经元个数分别为4和3的隐藏层。

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115173641.png)

首先给出网络中权重的定义，使用$w_{jk}^l$表示从 $l-1$ 层的 $k$ 个神经元到 $l$ 层的 $j$ 个神经元的连接权重。下图给出了网络中第二层的第四个神经元到第三层的第二个神经元的连接权重：

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115173828.png)

使用 $b_j^l$ 表示第 $l$ 层第 $j$ 个神经元的偏置，使用 $a_j^l$ 表示第 $l$ 层第 $j$ 个神经元的激活值，下图形象地展示了上述表示：

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181115173923.png)

$l$ 层第 $j$ 个神经元的激活值 $a_j^l$ 与 $l-1$ 层神经元的激活值通过如下方程关联
$$
\begin{equation}
a_j^l=\sigma(\sum_kw_{jk}^la_k^{l-1}+b_j^{l})
\end{equation}
$$
用向量形式表示如下
$$
\begin{equation}
a^{l}=\sigma(w^la^{l-1}+b^l)
\end{equation}
$$
在计算 $a^l$ 的过程中，引入中间量 $z^l=w^la^{l-1}+b^l$，称之为 $l$ 层神经元的带权输入。神经网络的前向传播方程如下
$$
\begin{equation}
z_j^l=\sum_kw_{jk}^la_k^{l-1}+b_j^l \\
a_j^l=\sigma(z_j^l)\\
l=2,3,...,L
\end{equation}
$$
用向量形式表示如下
$$
\begin{equation}
z^l=w^la^{l-1}+b^l \\
a^l=\sigma(z^l) \\
l=2,3,...,L
\end{equation}
$$
当 $l=2$ 时，$a_k^{(1)}$ 表示神经网络的输入。

### 损失函数

为了应用反向传播，损失函数需要满足如下两个假设

- 损失函数可以被表示为每个训练样本 $x$ 上的损失函数 $C_x$ 的均值，即
  $$
  C=\frac{1}{n}\sum_xC_x
  $$

- 损失函数可以被表示为关于神经网络输出的函数。

使用 $y_j,j=1,2,...,K$ 表示训练样本的真实类别，即若样本的真实类别为 $k$，则向量 $y$ 的第 $k$ 个维度上的数值为1，其他维度上的数值为0。$a_j^L,j=1,2,...,K$ 表示神经网络最后一层（输出层）的输出结果。本节主要介绍神经网络的两种损失函数：二次损失函数和交叉熵损失函数。

#### 二次损失函数

神经网络的二次代价函数具有如下形式
$$
\begin{eqnarray}
C&=&\frac{1}{2n}\sum_k||y(k)-a^L(k)||^2\\
&=&\frac{1}{2n}\sum_k\sum_j(y_j(k)-a_j^L(k))^2
\end{eqnarray}
$$
其中 $n$ 是训练样本的总数，$k$ 为训练样本的代号，$k=1,2,...,n$，$L$ 表示神经网络的层数。$y_j(k)$ 表示第 $k$ 个样本在第 $j$ 个维度上的真实类别取值，$a_j^L(k)$ 表示第 $k$ 个样本在第 $L$ 层（输出层）第 $j$ 个神经元的输出。单个样本的二次损失函数为
$$
\begin{eqnarray}
C&=&\frac{1}{2}||y-a^L||^2 \\
&=&\frac{1}{2}\sum_j(y_j-a_j^L)^2
\end{eqnarray}
$$
单个样本的二次损失函数对输出层激活值的梯度为
$$
\begin{equation}
\frac{\partial C}{\partial a^L}=a^L-y
\end{equation}
$$
二次损失函数可能存在着误差越大，收敛越缓慢的缺点。

#### 交叉熵损失函数

为了理解交叉熵的含义，给出如下的示例

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181116104023.png)

假设，现在需要训练一个包含若干输入变量 $x_1,x_2,...$ 的神经元，对应的权重为 $w_1,w_2,...$ ，偏置为 $b$ 。神经元的输出 $a=\sigma(z)$，其中 $z=\sum_jw_jx_j+b$ 是输入的带权和。定义该神经元的交叉熵代价函数如下
$$
\begin{equation}
C=-\frac{1}{n}\sum_x[ylog(a)+(1-y)log(1-a)]
\end{equation}
$$
其中 $n$ 为训练样本的总数，求和是在所有的训练输入 $x$ 上进行的，$y$ 是对应某个训练输入的目标输出。将交叉熵作为损失函数有如下两点原因

- 交叉熵是非负的，$C>0$。可以看出，上式中所有的独立项都为负的，因为对数函数的定义域为$(0,1)$，而求和前面有一个负号。
- 对于所有的训练输入 $x$，如果神经元的输出接近真实目标值，那么交叉熵将接近0。

对于输出层包含多个神经元的神经网络，其交叉熵损失函数定义如下
$$
\begin{equation}
C=-\frac{1}{n}\sum_k\sum_jy_j(k)loga_j^L(k)+(1-y_j(k))log(1-a_j^L(k))
\end{equation}
$$
单个训练样本的交叉熵损失函数为
$$
\begin{equation}
C=-\sum_jy_jloga_j^L+(1-y_j)log(1-a_j^L)
\end{equation}
$$
单个样本的交叉熵损失函数对输出层激活值的梯度为
$$
\begin{eqnarray}
\frac{\partial C}{\partial a_j^L}&=&-(\frac{y_j}{a_j^L}-\frac{1-y_j}{1-a_j^L}) \\
&=&-\frac{y_j-a_j^L}{a_j^L(1-a_j^L)}
\end{eqnarray}
$$
交叉熵损失函数一定程度上避免了二次损失函数权重学习缓慢的问题。

#### 两种损失函数的比较

假设对于一个训练输入 $x=1$ 的神经元，$y=0$ 为其目标输出。$w$ 和 $b$ 分别为神经元的权重和偏置，即有 $a=\sigma(z)$ ，其中 $z=wx+b$。二次损失函数定义为 $C=(y-a)^2/2$。使用链式法则得到 $C$ 对权重 $w$ 和偏置 $b$ 的偏导数如下
$$
\begin{equation}
\frac{\partial C}{\partial w}=(a-y)\sigma'(z)x=a\sigma'(z) \\
\frac{\partial C}{\partial b}=(a-y)\sigma'(z)=a\sigma'(z)
\end{equation}
$$
可以看出，损失函数对权重和偏置的偏导数 $\sigma'(z)$ 这一项有关，当激活函数为sigmoid函数时，若神经元此时的输出接近于1（偏离真实目标输出 $y=0$ )，sigmoid函数曲线变得相当平缓，则 $\sigma'(z)$ 的值会很小，从而导致 $\partial C/\partial w$ 和 $\partial C/\partial b$ 的值非常小。上述为二次损失函数学习缓慢的原因。

再来看交叉熵损失函数，对于输出层只有一个神经元的情况，考虑多个训练样本的损失函数 $C$ 对权重 $w_j$ 和偏置 $b$ 的偏导数，应用链式法则得到
$$
\begin{eqnarray}
\frac{\partial C}{\partial w_j}&=&-\frac{1}{n}\sum_x\frac{y-\sigma(z)}{\sigma(z)(1-\sigma(z))}\frac{\partial \sigma(z)}{\partial w_j}\\
&=&-\frac{1}{n}\sum_x\frac{y-\sigma(z)}{\sigma(z)(1-\sigma(z))}\sigma'(z)x_j\\
\end{eqnarray}\\
\frac{\partial C}{\partial b}=-\frac{1}{n}\sum_x\frac{y-\sigma(z)}{\sigma(z)(1-\sigma(z))}\sigma'(z)
$$
当激活函数为sigmoid函数时，根据 $\sigma(z)=1/(1+e^{-z})$ 可以得到 $\sigma'(z)=\sigma(z)(1-\sigma(z))$，带入上式可以得到
$$
\frac{\partial C}{\partial w_j}=\frac{1}{n}\sum_x(\sigma(z)-y)x_j\\
\frac{\partial C}{\partial b}=\frac{1}{n}\sum_x(\sigma(z)-y)
$$
可以看出，权重项和偏置项的学习受到 $\sigma(z)-y$ 的控制，更大的误差会产生更快的学习速度。在使用交叉熵损失函数时，$\sigma'(z)$ 被约去了，从而避免了二次损失函数中学习缓慢的问题。

### 反向传播算法

反向传播算法用于计算损失函数对于神经网络连接权重和偏置项参数的梯度。反向传播算法最初在1970年代被提及，但是直到 David Rumelhart、Geoffrey Hinton 和 Ronald Williams 的 "Learning representations by back-propagating errors" 论文出现后，人们才认识到这个算法的重要性。如今，反向传播算法已成为神经网络学习中的重要组成部分。本节主要介绍反向传播算法的四个基本方程。

#### Hadamard乘积

反向传播算法基于常规的线性代数运算，诸如向量加法、向量矩阵乘法等。特别地，假设 $s$ 和 $t$ 是两个维度相同的向量，使用 $s\bigodot t$ 表示两向量按元素的乘积，即 $(s\bigodot t)_j=s_jt_j$。示例如下
$$
\left[
\begin{matrix}
1 \\
2
\end{matrix}
\right]
\bigodot
\left[
\begin{matrix}
3 \\
4
\end{matrix}
\right]
=
\left[
\begin{matrix}
3 \\
8
\end{matrix}
\right]
$$
此运算被称作Hadamard乘积或Schur乘积。优秀的矩阵运算库通常会提供Hadamard乘积的快速实现，在实现反向传播的时候用起来很方便。

#### 四个基本方程

#####输出层误差方程

为了计算损失函数对权重和偏置的偏导数 $\partial C/\partial w_{jk}^l$ 和 $\partial C/\partial b_j^l$，首先引入一个中间量，$\delta_j^l$，称为第 $l$ 层第 $j$ 个神经元上的误差。反向传播将给出计算误差 $\delta_j^l$ 的方程，并将其关联到损失函数对权重和偏置的梯度计算中。误差 $\delta_j^l$ 定义如下
$$
\begin{equation}
\delta_j^l=\frac{\partial C}{\partial z_j^l}
\end{equation}
$$
其中 $z_j^l$ 为第 $l$ 层第 $j$ 个神经元的带权输入。根据链式法则，输出层误差计算公式为
$$
\begin{equation}
\delta_j^L=\frac{\partial C}{\partial a_j^L}\sigma'(z_j^L)
\end{equation}
$$
用向量形式表示如下
$$
\begin{equation}
\delta^L=\nabla_aC\bigodot \sigma'(z^L)
\end{equation}
$$
上式被称为输出层误差方程。

##### 误差反向传播方程

使用下一层的误差 $\delta^{l+1}$ 来表示上一层的误差 $\delta^l$ 
$$
\begin{equation}
\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot \sigma'(z^l)
\end{equation}
$$
其中 $(w^{l+1})^T$ 为第 $l+1$ 层权重矩阵 $w^{l+1}$ 的转置。下面给出该公式的证明
$$
\begin{equation}
\delta_j^l=\frac{\partial C}{\partial z_j^l}=\sum_k\frac{\partial C}{\partial z_k^{l+1}}\frac{\partial z_k^{l+1}}{\partial z_j^l}=\sum_k\delta_k^{l+1}\frac{\partial z_k^{l+1}}{\partial z_j^l}
\end{equation}
$$
其中第 $l+1$ 层第 $k$ 个神经元的带权输入 $z_k^{l+1}$ 可以表示为
$$
\begin{equation}
z_k^{l+1}=\sum_jw_{kj}^{l+1}a_j^l+b_k^{l+1}=\sum_jw_{kj}^{l+1}\sigma(z_j^l)+b_k^{l+1}
\end{equation}
$$
$z_k^{l+1}$ 对 $z_j^l$ 的偏导数为
$$
\begin{equation}
\frac{\partial z_k^{l+1}}{\partial z_j^l}=w_{kj}^{l+1}\sigma'(z_j^l)
\end{equation}
$$
将上式代入到 $\delta_j^l$ 的计算公式中可得
$$
\begin{equation}
\delta_j^l=\sum_k\delta_k^{l+1}w_{kj}^{l+1}\sigma'(z_j^l)
\end{equation}
$$
用向量形式表示如下
$$
\begin{equation}
\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot\sigma'(z^l)
\end{equation}
$$
上式被称为误差反向传播方程。

##### 权重方程

神经网络损失函数对权重参数 $w_{jk}^l$ 的梯度可以表示为
$$
\begin{equation}
\frac{\partial C}{\partial w_{jk}^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial w_{jk}^l}=\delta_j^l\frac{\partial z_j^l}{\partial w_{jk}^l}
\end{equation}
$$
第 $l$ 层第 $j$ 个神经元的带权输入 $z_j^l$ 可以表示为
$$
\begin{equation}
z_j^l=\sum_kw_{jk}^la_k^{l-1}+b_j^l
\end{equation}
$$
将上式代入到 $\partial C/\partial w_{jk}^l$ 中可以得到
$$
\begin{equation}
\frac{\partial C}{\partial w_{jk}^l}=\delta_j^la_k^{l-1}
\end{equation}
$$
用向量形式表示如下
$$
\begin{equation}
\frac{\partial C}{\partial w^l}=\delta^l(a^{l-1})^T
\end{equation}
$$
上式被称为权重方程。

##### 偏置方程

神经网络损失函数对偏置项 $b_j^l$ 的梯度可以表示为
$$
\begin{eqnarray}
\frac{\partial C}{\partial b_j^l}&=&\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial b_j^l}=\delta_j^l\frac{\partial z_j^l}{\partial b_j^l}\\
&=&\delta_j^l\frac{\partial (\sum_kw_{jk}^la_k^{l-1}+b_j^l)}{\partial b_j^l}\\
&=&\delta_j^l
\end{eqnarray}
$$
用向量形式表示如下
$$
\begin{equation}
\frac{\partial C}{\partial b^l}=\delta^l
\end{equation}
$$
上式被称为偏置方程。

综上所述，计算神经网络损失函数对权重和偏置项导数的四个基本方程如下
$$
\begin{equation}
\delta^L=\nabla_aC\bigodot \sigma'(z^L)\\
\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot \sigma'(z^l)\\
\frac{\partial C}{\partial w^l}=\delta^l(a^{l-1})^T\\
\frac{\partial C}{\partial b^l}=\delta^l
\end{equation}
$$

#### 算法描述

反向传播的四个基本方程给出了一种计算损失函数梯度的方法，显示地用算法描述如下

--------

1. **输入$x$**：为输入层设置对应的激活值 $a_j^1=x_j$。
2. **前向传播**：对每一层 $l=2,3,...,L$，计算相应的 $z^l=w^la^{l-1}+b^l$ 和 $a^l=\sigma(z^l)$。
3. **输出层误差**$\delta^L$： 计算向量 $\delta^L=\nabla_aC\bigodot \sigma'(z^L)$。
4. **反向传播**：对于每一层 $l=L-1,L-2,...,2$，计算相应的误差项 $\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot\sigma'(z^l)$。
5. **输出**：计算损失函数对权重和偏置的梯度 $\partial C/\partial w^l=\delta^l(a^{l-1})^T,\ \partial C/\partial b^l=\delta^l$。

--------

在实践中，通常将反向传播算法和诸如随机梯度下降这样的学习算法进行组合使用，给定一个大小为 $m$ 的小批量数据，以下为该小批量数据（batch）上的梯度下降学习算法

------

1. **输入训练样本的集合**。

2. **对每个训练样本** $x$：设置对应的神经网络输入激活 $a^{x,1}$，并执行下面的步骤

   - **前向传播**：对每个 $l=2,3,...L$，计算 $z^{x,l}=w^la^{x,l-1}+b^l$ 和 $a^{x,l}=\sigma(z^{x,l})$。
   - **输出层误差**$\delta^{x,L}$：计算向量 $\delta^{x,L}=\nabla_aC_x\bigodot \sigma'(z^{x,L})$。
   - **误差反向传播**：对于每个 $l=L-1,L-2,...,2$，计算 $\delta^{x,l}=((w^{l+1})^T\delta^{x,l+1})\bigodot \sigma'(z^{x,l})$。

3. **梯度下降更新**：对每个 $l=L,L-1,...,2$，更新权重项和偏置项如下
   $$
   \begin{equation}
   w^l\leftarrow w^l-\frac{\eta}{m}\sum_x\delta^{x,l}(a^{x,l-1})^T\\
   b^l\leftarrow b^l-\frac{\eta}{m}\sum_x\delta^{x,l}
   \end{equation}
   $$


------

在实践中实现随机梯度下降，还需要一个产生小批量数据（batch）的循环。具体代码实现详见后文。

### 常用优化方法

常用的优化方法包括两大类，一阶优化方法和二阶优化方法。在神经网络与深度学习中，常用的已有优化方法均为一阶梯度下降算法及其变形，很少使用 $Newton$ 或者 $Quasi \ Newton$ 等二阶方法，主要原因如下

- 使用二阶方法通常需要直接计算或近似估计 $Hessian$ 矩阵，对于深度神经网络而言，估计 $Hessian$ 矩阵的时间损耗使得其相比一阶方法在收敛速度上带来的优势被完全抵消。
- 深度学习中某些非线性网络层很难或不可能使用二阶方法优化。
- 二阶方法容易被鞍点（saddle points）吸引，难以达到局部或全局最优。$NIPS2014$ 有篇论文 "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization." 认为在高维情况下，神经网络优化最大的问题不是容易达到局部最优，而是容易被鞍点困住，然而很多鞍点存在于 $loss$ 较高的空间中。

以下主要介绍神经网络中常用的一阶优化算法，主要包括梯度下降算法（GD）、随机梯度下降算法（SGD）、Momentum算法、Nesterov Momentum算法、AdaGrad算法、RMSProp算法、Adam算法。

#### GD

在梯度下降学习算法中，每一步迭代都使用训练集的所有样本，定义 $\theta$ 为神经网络的参数集合，$f(x_i;\theta)$ 为神经网络中单个样本的前向传播输出，$C(f(x_i;\theta),y_i)$ 为单个样本的损失。梯度下降算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，学习速率 $\eta$ ，初始网络参数 $\theta$ 。

1. 提取训练集的所有样本 ${x_1,x_2,...,x_n}$，以及相应的真实输出 $y_1,y_2,...,y_n$。
2. 计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{n}\nabla_\theta\sum_{i=1}^{n}C(f(x_i;\theta),y_i)\\
\theta\leftarrow \theta-\eta g
\end{equation}
$$

--------

由于梯度下降算法中每次迭代都使用了训练集中的所有样本，因此当数据集很大时存在着收敛缓慢的问题。

#### SGD

SGD全名 stochastic gradient descent，即随机梯度下降。SGD算法每次迭代时随机抽取一批样本，并以此来更新权重和偏置参数。随机梯度下降算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，学习速率 $\eta$，初始网络参数 $\theta$。

1. 从训练集中生成若干批容量为 $m$ 的样本 ${x_1,x_2,...,x_m}$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^{m}C(f(x_i;\theta),y_i)\\
\theta\leftarrow \theta-\eta g \\
\end{equation}
$$

--------

相比于传统的梯度下降算法，随机梯度下降算法训练速度快，对于很大的数据集，也能以较快的速度收敛。但随机梯度下降算法每一次迭代的梯度受到抽样的影响，可能含有比较大的噪声。

#### Momentum

随机梯度下降算法所采用的路径可能沿着局部最优收敛方向摆动，Momentum算法则相当于给参数更新增加了惯性，以减小SGD算法中存在的振荡问题。Momentum算法考虑了过去的梯度方向以平滑更新，其引入了一个新的速率变量 $v$，$v$ 是之前梯度的累加，但每回合都有一定的衰减。Momentum算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，学习速率 $\eta$，初始速率 $v$，动量衰减参数 $\alpha$，初始网络参数 $\theta$。

1. 从训练集中生成若干批容量为 $m$ 的样本 ${x_1,x_2,...,x_m}$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^mC(f(x_i;\theta),y_i)\\
v\leftarrow \alpha v-\eta g\\
\theta\leftarrow \theta+v
\end{equation}
$$

--------

直观上来看，当目前的梯度方向和之前方向相反时，结合之前的梯度，就会减小参数更新振荡的幅度。当目前的梯度方向和之前方向相同时，则加快收敛速度。若收敛过程中梯度方向都大致相似，其收敛速度的加快与参数 $\alpha$ 的关系为
$$
\begin{equation}
v=-\frac{\eta g}{1-\alpha}
\end{equation}
$$
即若动量衰减参数 $\alpha=0.99$，其收敛速度为 SGD 的大约100倍。下图粗略表示了 Momentum 和 SGD 收敛的方向。

![](D:\DataDoc\deep_learning\神经网络与深度学习实战\微信截图_20181119192920.png)

图中红色箭头显示了使用动量梯度下降的下降方向，蓝点表示随机梯度下降的下降方向，可以看出 Momentum 算法修正了 SGD 中大幅度振荡收敛的问题。

#### Nesterov Momentum

Nesterov Momentum 是 Momentum 的变种，该方法有机会解决 Momentum 方法越过最小值的问题。其改进点在于，计算梯度前，先用当前速度更新一次参数，再用更新后的临时参数计算梯度。Nesterov Momentum算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，学习速率 $\eta$，初始速率 $v$，动量衰减参数 $\alpha$，初始网络参数 $\theta$。

1. 从训练集中生成若干批容量为 $m$ 的样本 $x_1,x_2,...,x_m$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^{m}C(f(x_i;\theta+\alpha v),y_i)\\
v\leftarrow \alpha v - \eta g\\
\theta\leftarrow \theta+v
\end{equation}
$$

--------

与 Momentum 方法唯一区别的一点是在估算梯度 $g$ 时，网络参数变成了 $\theta+\alpha v$ 而不是之前的 $\theta$。

#### AdaGrad

前述方法中，对网络参数的每一个维度 $\theta_i$ 都采用了相同的学习率 $\eta$，AdaGrad 算法能够在训练中自动地对学习率进行调整，其自适应地为各个方向的梯度分配不同的学习率。AdaGrad 算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，全局学习速率 $\eta$，数值稳定量 $\epsilon(10^{-7})$，初始网络参数 $\theta$。

中间变量：梯度累计量 $r$（初始化为0）

1. 从训练集中生成若干批容量为 $m$ 的样本 $x_1,x_2,...,x_m$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta \sum_{i=1}^{m}C(f(x_i;\theta),y_i)\\
r\leftarrow r + g\bigodot g\\
\theta\leftarrow \theta - \frac{\eta}{\epsilon+\sqrt{r}}\bigodot g
\end{equation}
$$

--------

基于如下假设：在训练最初阶段，模型参数距离损失函数最优解较远，随着更新次数的增加，参数会越来越接近最优解。因此在 AdaGrad 算法中，各个梯度方向的学习率会越来越小。AdaGrad 算法会使得梯度较大方向上的更新越来越慢，梯度较小方向上的更新也会越来越慢，但变慢的速率明显小于梯度大的方向，从而使得各个方向上的梯度更新趋于一致。AdaGrad 算法较为适用于稀疏数据集的训练。

#### RMSProp

RMSProp 是 AdaGrad 的变种，其通过引入一个衰减系数，使得梯度累计量 $r$ 在每次更新过程中都衰减一定比例。RMSProp 算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，全局学习速率 $\eta$，数值稳定量 $\epsilon(10^{-7})$，衰减速率 $\rho$，初始网络参数 $\theta$。

 中间变量：梯度累计量 $r$（初始化为0）

1. 从训练集中生成若干批容量为 $m$ 的样本 $x_1,x_2,...,x_m$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^mC(f(x_i;\theta),y_i)\\
r\leftarrow \rho r+(1-\rho)g\bigodot g\\
\theta\leftarrow \theta-\frac{\eta}{\epsilon+\sqrt{r}}\bigodot g
\end{equation}
$$

--------

RMSProp 可以缓解 AdaGrad 中学习率下降较快的问题。也有将 RMSProp 和 Nesterov Momentum 结合起来使用的算法，其每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，全局学习速率 $\eta$，初始速率 $v$，梯度累计量衰减速率 $\rho$，动量衰减参数 $\alpha$，数值稳定量 $\epsilon(10^{-7})$，初始网络参数 $\theta$。

中间变量：梯度累计量 $r$（初始化为0）

1. 从训练集中生成若干批容量为 $m$ 的样本 $x_1,x_2,...,x_m$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
\bar{\theta}\leftarrow \theta+\alpha v\\
g\leftarrow \frac{1}{m}\nabla_{\bar{\theta}}\sum_{i=1}^{m}C(f(x_i;\theta),y_i)\\
r\leftarrow \rho r+(1-\rho)g\bigodot g\\
v\leftarrow \alpha v - \frac{\eta}{\epsilon+\sqrt{r}}\bigodot g\\
\theta\leftarrow \theta+v
\end{equation}
$$

--------

RMSProp-Nesterov 算法已经被证明是一种有效且实用的深度神经网络优化算法。

#### Adam

Adam（Adaptive Moment Estimation）本质上是带有动量项的 RMSProp，它利用梯度的一阶矩估计和二阶矩估计动态调整各个梯度方向的学习率。Adam 的优点主要在于经过偏置校正后，每一次迭代的学习率都有一个确定范围，使得参数更新比较平稳。Adam 算法的每次迭代更新过程如下

--------

训练样本 $x$，真实输出 $y$，学习速率 $\eta$，数值稳定量 $\epsilon$，一阶动量衰减系数 $\rho_1$，二阶动量衰减系数 $\rho_2$，初始网络参数 $\theta$。

参数常用取值：$\epsilon=10^{-7},\rho_1=0.9,\rho_2=0.999$

中间变量：一阶动量 $s$，二阶动量 $r$，初始化均为0

1. 从训练集中生成若干批容量为 $m$ 的样本 $x_1,x_2,...,x_m$，以及相应的真实输出 $y_1,y_2,...,y_m$。
2. 对于每一批训练样本，计算梯度并更新参数：

$$
\begin{equation}
g\leftarrow \frac{1}{m}\nabla_\theta\sum_{i=1}^{m}C(f(x_i;\theta), y_i)\\
s=\rho_1s+(1-\rho_1)g\\
r=\rho_2r+(1-\rho_2)g\bigodot g\\
\bar{s}=\frac{s}{1-\rho_1}\\
\bar{r}=\frac{r}{1-\rho_2}\\
\theta=\theta-\eta\frac{\bar{s}}{\epsilon+\sqrt{\bar{r}}}
\end{equation}
$$

--------

Adam 算法目前是深度神经网络优化中最常使用的算法。

### 简易神经网络库

```c++
class Solution
{
    public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
    {
        if (nums1.size() > nums2.size())
        {
            return findMedianSortedArrays(nums2, nums1);
        }
        int m = nums1.size();
        int n = nums2.size();
        int iMin = 0;
        int iMax = nums1.size();
        int halfLen = (m + n + 1) / 2;
        while (iMin <= iMax)
        {
            int i = (iMin + iMax) / 2;
            int j = halfLen - i;
            if (i < iMax && nums2[j - 1] > nums1[i])
            {
                iMin = i + 1;
            }
            else if (i > iMin && nums1[i - 1] > nums2[j])
            {
                iMax = i - 1;
            }
            else
            {
                int maxLeft = 0;
                int minRight = 0;
                if (i == 0)	{ maxLeft = nums2[j - 1]; }
                else if (j == 0)	{ maxLeft = nums1[i - 1]; }
                else	{ maxLeft = max(nums1[i - 1], nums2[j - 1]); }
                if ((m + n) % 2 == 1)	{ return maxLeft; }
                if (i == m)	{ minRight = nums2[j]; }
                else if (j == n)	{ minRight = nums1[i]; }
                else	{ minRight = min(nums1[i], nums2[j]); }
                if ((m + n) % 2 == 0)	{ return (maxLeft + minRight) / 2.0; }
            }
        }
    }
};
```



