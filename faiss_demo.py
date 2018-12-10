###################### load packages ####################
import numpy as np
import faiss
from faiss import normalize_L2

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

###################### build index ####################
index = faiss.IndexFlatL2(d)
print(index.is_trained)

######## add vectors to the index ########
index.add(xb)
print(index.ntotal)

###################### search result L2 ####################
###### 4 nearest neighbors ######
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

###################### search result Cosine similarity ####################
####### vector ids #######
ids = np.arange(nb)
ids = np.array(ids, dtype=np.int64)

###### build index #######
index = faiss.IndexFlatIP(d)

###### normalize xb 即: xb / np.sqrt((xb**2).sum()) #######
normalize_L2(xb)

###### index Map #######
index2 = faiss.IndexIDMap(index)

###### add vectors to the index ######
index2.add_with_ids(xb, ids)

###### search result ######
D, I = index2.search(xb[:5], k)