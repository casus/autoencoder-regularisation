import time

import numpy as np

from minterpy.transformation import Transformer

np.random.seed(23232323)
# m,n,lp_degree
input_para = (2, 3, 2)


startFull = time.time()

start = time.time()
test_tr = Transformer(*input_para)

print(test_tr.exponents)


lagrange_coefs = np.zeros(test_tr.N)
base_coefs = np.random.uniform(-10, 10, test_tr.N)
start = time.time()
for i in np.arange(test_tr.N):
    temp_lag = np.ones(test_tr.N)
    for j in np.arange(test_tr.N):
        for d in np.arange(test_tr.m):
            temp_lag[j] = test_tr.tree.grid_points[d, i] * test_tr.exponents[d, j]
        temp_lag[j] *= base_coefs[j]
    lagrange_coefs[i] = np.sum(temp_lag)

canon = test_tr.transform_l2c(lagrange_coefs)

print("---- results ----")
abs_err = np.abs(base_coefs - canon)
print("max abs_err", abs_err.max())
