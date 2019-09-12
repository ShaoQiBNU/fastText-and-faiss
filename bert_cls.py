#!/usr/bin/env python
# coding=utf-8
import json
import sys
import numpy as np
import mkl
mkl.get_max_threads()
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

    ############# query里每个字的embedding，做concat #############
    for i in data['features']:
        token = token + i['token']

    for j in data['features'][0]['layers']:
        value.extend(j['values'])

    ############# 结果保存 #############
    values.append(value)
    tokens.append(token)
    line_index.append(data['linex_index'])

    '''
    if data['linex_index'] % 10000 ==0:
        print(data['linex_index'])

    print(data['linex_index'], token, length, sep="\t")
    '''


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

# kmeans.centroids 聚类中心
D, I = kmeans.index.search(query_values, 1)

i = 0
for center in np.nditer(I):
    print(i, tokens[i], center, sep="\t")
    i = i + 1

'''
index = faiss.IndexFlatL2(d)
index.add(query_values)
D, I = index.search(kmeans.centroids, 2)
'''