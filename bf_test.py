import sys
sys.path.append('/home/ubuntu/projects/dem/topdot11/build')
from topdot import _topdot, _bf_dense

import numpy as np
from time import time

np.random.seed(123)

D = np.random.uniform(0, 1, (1000, 720)).astype(np.float64)
D = D / np.sqrt((D ** 2).sum(axis=-1, keepdims=True))

Q = D[:,:120]

D = np.ascontiguousarray(D)
Q = np.ascontiguousarray(Q)

sim = np.zeros((Q.shape[0], D.shape[0]))
t = time()
_bf_dense(D.shape[0], D.shape[1], Q.shape[0], Q.shape[1], D.ravel(), Q.ravel(), sim)
print(time() - t)