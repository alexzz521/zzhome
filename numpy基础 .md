 list可以存储不同类型的元素，array只能存储同一类型的数据，numpy.ndarray也只能存同类的，操作和list和array基本一致 

# np创建数组

## 1.直接np.array([1,2,3])

## 2.通过zeros，ones，full创建

np.zeros(10),创建有10个0的数组
np.zeros(10).dtype 查看数据类型，默认是float
np.zeros(10，dtype=int)也可在创建的时候指定

np.ones()创建全1

np.full(shape=(3,3),fill_value=666),创建内容为666的3*3矩阵

## 3.通过arange或linspace创建，在指定区域取指定个数或者取等长

np.arange(0,10),创建一个从0开始到9结束的数组，后面还能加个数表示步长，也可以省略前面的默认从0开始

np.linspace(0,20,10) 从0-20等长取10个元素，0和20包含

## 4.通过random创建，随机创建

np.random
np.random.randint(0,10),在0-10不包含10中去一个随机数
np.random.randint(0,10,size=10)生成一个大小为10，范围在0-10不包含10的数组中的随机数组，如果size=（3,3）就是生成3*3随机矩阵
np.random.seed(666) 设置随机数种子，如果随机数种子一致，那就会生成相同的随机数数列
np.random.random() 创建随机浮点数
np.random.random(10) 创建10个随机浮点数
np.random.random((3,3)) 创建3*3随机浮点数矩阵
np.random.normal()创建一个服从标准正态分布的随机浮点数
np.random.normal(10,100)，指定均值为10，方差为100的分布里的随机浮点数，第三个参数为大小

# np数组的基本操作

## 1.通过reshape改变原数组大小

X=np.arange(15).reshape(3,5)  前面是生成0-15步长为1的数组，后面是将数组变成3*5的矩阵
！！！调用reshape是不会改变自身的，需要拿变量承接，reshape中的参数如果为-1，表示自动分配比如
B=X.reshape(3,-1) 表示我把它分成3行，每行多少元素不管，当然如果不能被3整除就寄了
B=array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])

## 2.查看维数，大小

X.ndim 查看是几维的数组
X.shape 返回一个元组，表示有几个维度，大小多少
X.size 返回大小

## 3.多维数组访问和拿取

X[2,2],尽量不用X[2][2]
也可以使用切片
X[:5,:]
切片中也能加入步长
X[::2,::2]
X[::-1,::-1]这就实现了矩阵的反转
使用numpy.ndarray子矩阵的操作会影响原来矩阵
如果想不被影响，则要使用copy()
subX=X[:2,:3].copy()这样对subX操作就不会影响X了

## 4.数组的拼接利用concatenate，vstack，hstack

np.concatenate([x,y,z,a])  实现数组的合并，[]里能有多个
np.concatenate([A,A],axis=1) 表示按照列的方向拼接，axis表示轴，默认为0，向一维拼接
A=np.array([[1,2,3],[3,2,1]])
array([[1, 2, 3],
       [3, 2, 1]])
np.concatenate([A,A])  默认一维拼接，也就是axis为0
array([[1, 2, 3],
       [3, 2, 1],
       [1, 2, 3],
       [3, 2, 1]])
np.concatenate([A,A],axis=1) 表示按照列的方向拼接
array([[1, 2, 3, 1, 2, 3],
       [3, 2, 1, 3, 2, 1]])
Z=np.array([1,2,3])
要将Z加入A中，需要给Z增加维度，然后concatenate
如 np.concatenate([A,Z.reshape(1,-1)]) 这样就能实现添加
array([[1, 2, 3],
       [3, 2, 1],
       [1, 2, 3]])
 也可以使用vstack，np.vstack([A,Z]),竖直方向堆叠
 array([[1, 2, 3],
       [3, 2, 1],
       [1, 2, 3]])
np.hstack([]) 水平方向堆叠
使用vstack和hstack前提是能够堆叠

## 5.数组的分割操作

np.split(X,[a,b]) ，根据a,b的位置将X分成三份，左包含，右不包含

对于矩阵A，np.split(A,[2])，根据行将A分成两份
比如A=[[1,2,3,4],[1,2,3,4],[1,2,3,4]，[1,2,3,4]]
通过np.split(A,[2])，得到[[1,2,3,4],[1,2,3,4]]和[[1,2,3,4]，[1,2,3,4]]
可以通过加入axis参数，将其改为根据列分，axis调整轴,比如np.split(A,[2],axis=1)
结果就变成了[[1,2],[1,2]，[1,2]，[1,2]]和[[3,4],[3,4]，[3,4]，[3,4]]
也可以通过np.vsplit(A,[2])直接在垂直方向切，也就是沿着行切，hsplit在水平方法分割，也就是沿着列方向切

## 6.具体计算操作

### 数和矩阵的运算
原生list*2，不是对list中的每个值乘2，而是list重复两次，比如list=[1,2]，变成了[1,2,1,2]
但是在numpy中，则是直接对值进行操作，基本的代数运算基本都是对数进行操作，比如A=np.array([1,2])
2*A，就是[2,4],占用的时间也少，效率较高。
np.abs(A)求绝对值
np.sin(A)求正弦
np.exp(A)求以e为底的指数
np.power(n,A),求以n为底的指数
np.log(A),求以e为底的对数
np.log10(A),求以10为底的对数

### 矩阵和矩阵的运算
上面的对数的运算，在numpy中对矩阵的运算也很简单，也是直接的数进行操作，一对一操作，对应元素做相应运算
要想得到正常的矩阵乘法，调用A.dot(B)这样调用的就是矩阵的乘法

A.T得到矩阵的转置

### 向量和矩阵的运算
向量和矩阵的每一行做计算，比如向量v=np.array([1,2]),对A=np.array([[0,1],[1,2]])
如果v+A结果就是[[1,3],[2,4]]。
其实等效于np.vstack([v]*A.shape[0])+A
也可以直接利用tile实现叠加，比如np.tile(v,(2,1))这里的v表示要叠的向量，2表示行方向上加叠两层，1表示列方向上叠一次
v*A也是一样，对应元素相乘，如果要实现正常的矩阵乘法可以用dot，在这里v.dot(A)可以，也可以A.dot(v),因为会自动调整v是行向量还是列向量方便相乘

### 矩阵的逆
可以通过invA=np.linalg.inv(A),求A的逆矩阵，
invA.dot(A)得到单位矩阵
A.dot(invA)也可以得到单位矩阵
只有方阵才有逆矩阵
没有逆矩阵的可以求伪逆
pinv*B=np.linalg.pinv(B)
原本8*2会得到2*8
pinvB.dot(B)结果是一个单位矩阵
B.dot(pinvB)结果也是一个单位矩阵

### 聚合运算
首先原本就可以通过sum(L)求和，但是numpy还是提供了np.sum(A)的算法，后者效率高
np.min()
np.max()求最大最小，也可以直接向量名.max()，或者向量名.min(),推荐前者
np.sum()如果是对矩阵进行求和，可以添加axis参数实现各行各列的求和
例如np.sum(A,axis=0)就是沿着行进行求和，得到各列的数据，如果为1就是沿着列进行求和，得到各行的数据
np.prod(A)所有元素相乘
np.mean()求均值
np.median()求中位数
np.percentile(A,q=50) 用来求百分位，一般q的值是0,25,50,75,100,0表示最小，50表示中位数，100最大值，25和75位四分位点，通过这四个点可以估算样本的分布
np.var()方差
np.std()求标准差


## 7.排序与索引
np.argmin(A)返回A中最小值所在位置的索引值
np.argmax(A)最大值的索引

x=np.arange(16)
np.random.shuffle(x) 这一步表示对x进行乱序处理

np.sort(x) 对x进行排序，x本身没变
如果要x本身变，那就x.sort()

对于二维矩阵，用sort默认是对行进行排序，对每行进行排序，其实就是沿着列的方向进行操作，所以axis默认为1
如果改成axis=0，那么每一列都会进行排序

np.argsort(x)这样可以得到排完序后的原来的索引，
比如原本x是 [2,5,7,3,4,6],使用np.argsort(x)后变成了[0,3,4,1,5,2]

np.partition(x,3),返回的前面的都比3小，后面都比3大，这样可以找到对某个值小或者大的数

np.argpartition(x,3)返回的是原来的索引

对矩阵操作也是一样的，只是矩阵可以通过调整axis来明确是沿着行还是列进行操作

## 8.Fancy Indexing

ind =[a,b,c]
x[ind]这样就能得到x[a],x[b],x[c]的元素

ind=np.array([[a,b],[c,d]])
x[ind]这样会得到[[x[a],x[b]],[x[c],x[d]]]也就是取出矩阵
对于矩阵，可以设置两个向量来存储索引
row=np.array([a,b,c])
col=np.array([d,e,f])
x[row,col] 就能从矩阵中取得自己想要的x[a,d],x[b,e],x[c,f]
也可以只放一个x[0,col]，或者有切片x[:2,col]

如果里面的布尔值，例如col=[TRUE,FALSE,TRUE,FALSE]表明对第0和第2列感兴趣

## 9.numpy.array的比较
x=np.array([0,1,2,3,4])
x<3
这样会得到一个bool向量
[true,true,ture,false,false]
这样就能配合Fancy Indexing得到需要的结果即x[x<3]
也能通过其它进行配合，比如np.sum(x<3),或者np.count_nonzero(x<3).
np.any()里面只要有一个满足就是true
np.all()里面全部满足才是true
其它>=,==,!=类似，甚至可以通过书写代数式，比如2*x==x-6，np.sum(x % 2==0)，
这种比较也能用于矩阵中 ,在矩阵中通过axis对每行，每列进行操作。
np.sum((x>3)&(x<5))
np.sum((x<5)|(x>7))
np.sum(-(x==1))
x[x[:,3]%3==0,:],这样可以去除第4列中能被3整除的元素所在行的所有数据













