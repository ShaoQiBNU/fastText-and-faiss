# 一. faiss相关资料

https://blog.csdn.net/kanbuqinghuanyizhang/article/details/80774609


# 二. faiss检索实例

## 1. 构建训练数据和测试数据

> 构建shape为[100000, 64]的训练数据xb和shape为[10000, 64]的查询数据xq，代码如下：

```python
###################### load packages ####################
import numpy as np
import faiss

###################### generate data ####################
###### dimension ######
d = 64

###### database size ######
nb = 100000

###### nb of queries ######
nq = 10000

###### make reproducible ######
np.random.seed(1234)

###### 训练数据 ######
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

###### 查询数据 ######
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
```

## 2. 构建索引

> faiss创建索引对向量预处理，提高查询效率，提供了多种索引方法，这里选择最简单的暴力检索L2距离的索引：IndexFlatL2。创建索引时必须指定向量的维度d。大部分索引需要训练的步骤，IndexFlatL2跳过这一步，代码如下：

```python
###################### build index ####################
index = faiss.IndexFlatL2(d)
print(index.is_trained)

######## add vectors to the index ########
index.add(xb)
print(index.ntotal)
```

## 3. 查找相似向量

>当索引创建好并训练(如果需要)之后，就可以执行add和search方法了。add方法一般添加训练时的样本，search就是寻找相似向量了。一些索引可以保存整型的ID，每个向量可以指定一个ID，当查询相似向量时，会返回相似向量的ID及相似度(或距离)。如果不指定，将按照添加的顺序从0开始累加，其中IndexFlatL2不支持指定ID。代码如下：

```python
###################### search result ####################
###### 4 nearest neighbors
k = 4

# sanity check
D, I = index.search(xb[:5], k)

print(D)
print(I)

# actual search
D, I = index.search(xq, k)

# neighbors of the 5 first queries
print(I[:5])

# neighbors of the 5 last queries
print(I[-5:])
```

> 上面代码中，定义返回每个需要查询向量的最近4个向量。查询返回两个numpy array对象`D`和`I`。`D`表示与相似向量的距离(distance)，`I`表示相似用户的ID。其中与其最相似的向量为其自身，所以`D`的第一列为0，`I`的第一列为查询向量自身的编号，结果如下：

```
[[0.        7.1751733 7.207629  7.2511625]
 [0.        6.3235645 6.684581  6.7999454]
 [0.        5.7964087 6.391736  7.2815123]
 [0.        7.2779055 7.5279865 7.6628466]
 [0.        6.7638035 7.2951202 7.3688145]]
[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]
```

# 三. 向量余弦相似度查找

> faiss提供了向量余弦相似度查找功能，具体方法如下：

```python
###################### search result Cosine similarity ####################
####### vector ids #######
ids = np.arange(nb)
ids = np.array(ids, dtype=np.int64)

###### build index #######
index = faiss.IndexFlatIP(d)

###### normalize xb 即: xb / np.sqrt((xb**2).sum()) #######
normalize_L2(xb)

###### index Map #######
'''
该函数的作用如下：
This index encapsulates another index and translates ids when adding and searching. It maintains a table with the mapping.

https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing
'''

index2 = faiss.IndexIDMap(index)

###### add vectors to the index ######
index2.add_with_ids(xb, ids)

###### search result ######
D, I = index2.search(xb[:5], k)
```


# 四. 向量聚类

> faiss做k-means聚类实例，以bert的最后一层输出向量为例，提取每个query的cls向量，然后聚类，输出每个query的聚类中心，代码如下：

```python
#!/usr/bin/env python
# coding=utf-8
import json
import sys
import numpy as np
import faiss

values = []
tokens = []
line_index = []

############# 逐行读入json文件，并解析 #############
for line in sys.stdin:
    line = line.strip()
    data = json.loads(line)

    token = ''
    value = []

    ############# 提取query #############
    for i in data['features']:
        token = token + i['token']

    ############# 提取query里的cls向量 #############
    for j in data['features'][0]['layers']:
        value.extend(j['values'])

    ############# 结果保存 #############
    values.append(value)
    tokens.append(token)
    line_index.append(data['linex_index'])

############# 数据转换类型 ##############
query_values = np.array(values)
query_values = query_values.astype('float32')

############# k-means 聚类 ##############
ncentroids = 1000
niter = 20
verbose = False
d = 768

kmeans = faiss.Kmeans(d, ncentroids, niter, verbose)
kmeans.train(query_values)


############# 获取每个query的聚类中心 ###############
D, I = kmeans.index.search(query_values, 1)

i = 0
for center in np.nditer(I):
    print(i, tokens[i], center, sep="\t")
    i = i + 1
```
