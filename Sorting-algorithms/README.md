# 排序算法
常见的排序算法有冒泡排序、选择排序、插入排序、归并排序、快速排序、堆排序、希尔排序、计数排序、桶排序和基数排序。
## 冒泡排序
冒泡排序（Bubble-sort）是一种简单的排序算法。它重复地走访要排序的数列，一次比较两个相邻元素，如果它们的大小顺序错误就把它们进行交换。这个算法的名字由来是因为越大的元素会经由交换慢慢“浮”到数列的顶端。
时间复杂度为 
<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=n^2" title="T(n)=n^2" /></a>

## 选择排序
选择排序（Selection-sort）是一种简单直观的排序算法。它的工作原理：首先在未排序序列中找到最大元素，存放到排序序列的最后位置，然后再从剩余未排序元素中继续寻找最大元素放到序列的末尾。以此类推，直到所有元素均排序完毕。时间复杂度为<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=n^2" title="T(n)=n^2" /></a>。

## 插入排序
插入排序（Insertion-sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并(插入。插入排序在实现上，通常采用in-place排序（即只需用到O(1)的额外空间），因而在从后向前扫描过程中，需要反复把已排序元素和新元素进行交换。时间复杂度为<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=n^2" title="T(n)=n^2" /></a>。

## 归并排序
归并排序（Merge-sort）是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个典型应用。归并排序先使每个子序列有序，再使子序列段间有序。若将两个有序序列合并成一个有序序列，称为2-路归并。时间复杂度：\\
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;T(n)&=2T(n/2)&space;&plus;&space;O(n)&space;\\&space;&=2^2T(n/2^2)&space;&plus;&space;O(2n)&space;\\&space;&=......\\&space;&=2^{log_2n}T(1)&plus;O(nlog_2n)\\&space;&=O(nlog_2n)&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;T(n)&=2T(n/2)&space;&plus;&space;O(n)&space;\\&space;&=2^2T(n/2^2)&space;&plus;&space;O(2n)&space;\\&space;&=......\\&space;&=2^{log_2n}T(1)&plus;O(nlog_2n)\\&space;&=O(nlog_2n)&space;\end{aligned}" title="\begin{aligned} T(n)&=2T(n/2) + O(n) \\ &=2^2T(n/2^2) + O(2n) \\ &=......\\ &=2^{log_2n}T(1)+O(nlog_2n)\\ &=O(nlog_2n) \end{aligned}" /></a>

## 快速排序
快速排序的基本思想：通过一趟排序将待排序序列分割成独立的两部分，其中一部分记录的序列值均比另一部分的序列值小。分别对两部分子序列继续进行排序，以达到整个序列有序。时间复杂度
<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=T(x)&plus;T(n&space;-&space;x)&space;&plus;&space;O(n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=T(x)&plus;T(n&space;-&space;x)&space;&plus;&space;O(n)" title="T(n)=T(x)+T(n - x) + O(n)" /></a>
其中x和n-x为两部分子序列的长度。最佳情况下当x=n/2时，<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=nlog_2n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=nlog_2n" title="T(n)=nlog_2n" /></a>；最差情况下当x=1时，<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=n^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=n^2" title="T(n)=n^2" /></a>。

## 堆排序
堆排序（Heap-sort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子节点的键值或索引总是小于（大于）它的父节点。时间复杂度<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=nlog_2n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=nlog_2n" title="T(n)=nlog_2n" /></a>。

## 希尔排序
希尔排序是希尔（Donald Shell）于1959年提出的一种排序算法。希尔排序也是一种插入排序，它是简单插入排序经过改进之后的一个更高效版本，也称为缩小增量排序，它与插入排序的不同之处在于会优先比较距离较远的元素。希尔排序是把序列按照{n/2, n/2/2, ..., 1}的间隔距离分组，对每组使用直接插入排序算法排序。随着增量逐渐减少，每组包含的序列值越来越多，当增量减至1时，整个序列恰被分成一组，算法终止。希尔排序的执行时间依赖于增量序列的选择，通常情况下时间复杂度为<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)=O(n^{(1.3-2)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)=O(n^{(1.3-2)})" title="T(n)=O(n^{(1.3-2)})" /></a>。

## 计数排序
计数排序是一种稳定的排序算法。计数排序使用一个额外的数组C，其中第i个元素是待排序数组A中值等于i的元素个数。然后根据数组C来将A中的元素排到正确的位置。它只能对整数进行排序。时间复杂度<a href="https://www.codecogs.com/eqnedit.php?latex=T(n)&space;=&space;O(n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T(n)&space;=&space;O(n)" title="T(n) = O(n)" /></a>。

## 桶排序
桶排序是计数排序的升级版。假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶内分别使用直接插入排序法进行排序，最后将所有不是空的桶里排好序的数据拼接起来。对N个元素的序列进行桶排序的时间复杂度分为两个部分：\
(1) 循环计算序列中每个元素的桶映射函数，这个时间复杂度为O(n)；\
(2) 利用先进的比较排序算法(直接插入排序)对每个桶内的所有数据进行排序，其时间复杂度为<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_in_ilog_2{n_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_in_ilog_2{n_i}" title="\sum_in_ilog_2{n_i}" /></a>，其中<a href="https://www.codecogs.com/eqnedit.php?latex=n_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n_i" title="n_i" /></a>为第i个桶的数据量。\
第(2)部分是桶排序性能好坏的决定因素。尽量增加桶的数量、减少桶内数据量是提高效率的唯一办法。因此需要尽量做到以下两点：\
a. 映射函数能够将n个数据平均分配到m个桶中，这样每个桶内有n/m个数据量；\
b. 尽量增加桶的数量。极限情况下每个桶只能得到一个数据，这样可以完全避开桶内数据的“比较”排序操作。在数据量巨大的情况下，映射函数会使得桶集合的数量巨大，空间浪费严重。\
对于n个待排数据，m个桶，平均每个桶n/m个数据的桶排序平均时间复杂度为：\
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;T(n)&=O(n)&plus;O(m\times&space;n/m\times&space;log_2&space;(n/m))&space;\\&space;&=O(n&plus;n(log_2n-log_2m))&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;T(n)&=O(n)&plus;O(m\times&space;n/m\times&space;log_2&space;(n/m))&space;\\&space;&=O(n&plus;n(log_2n-log_2m))&space;\end{aligned}" title="\begin{aligned} T(n)&=O(n)+O(m\times n/m\times log_2 (n/m)) \\ &=O(n+n(log_2n-log_2m)) \end{aligned}" /></a> \
当m=n时，即极限情况下的每个桶内只有一个数据，桶排序的最好效率能达到O(n)。\
桶排序的平均时间复杂度为线性的O(n+c)，其中<a href="https://www.codecogs.com/eqnedit.php?latex=c=n(log_2n-log_2m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?c=n(log_2n-log_2m)" title="c=n(log_2n-log_2m)" /></a>。如果相对于同样的n，桶数量m越大，其效率越高，最好的时间复杂度可以达到O(n)。桶排序的空间复杂度为O(n+m)，如果输入数据非常庞大，而桶的数量也非常多，则空间代价十分昂贵。

## 基数排序
基数排序是按照低位先排序，然后收集；再按照高位排序，然后再收集；以此类推，直到最高位。时间复杂度为O(kn)，n为待排序数组的长度，k为数组中最大数的位数。基数排序只能对整数进行排序。



