import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

row = np.array([i*1000 for i in range(10000)])
col = np.array([i*1000 for i in range(10000)])
dat = np.array([i*i    for i in range(10000)])

m = csr_matrix((dat, (row, col)))

m = m.dot(m)

print(type(m[2000][0]))

row1 = [0, 1, 2]
col1 = [0, 1, 2]
dat1 = [1, 2, 3]
n = csr_matrix((dat1, (row1, col1)))

n = n.dot(n)

a = csr_matrix(np.zeros([100000, 1]))
b = csr_matrix(np.zeros([100000, 1]))
print(a.dot(b.T).sum())
