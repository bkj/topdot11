#!/usr/bin/env python

"""
    test.py
"""

from __future__ import print_function, division

import sys
import json
import argparse
import numpy as np
from time import time
from scipy import sparse

sys.path.append('build')
import topdot

def run_topdot(A, B, k, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    
    num_rows = A.shape[0]
    
    I = np.empty(num_rows * k, dtype=np.int32)
    D = np.empty(num_rows * k, dtype=A.dtype)
    
    topdot._topdot(
        n_row=np.int32(num_rows),
        n_col=np.int32(B.shape[1]),
        
        Ap=np.asarray(A.indptr, dtype=np.int32),
        Aj=np.asarray(A.indices, dtype=np.int32),
        Ax=A.data,
        
        Bp=np.asarray(B.indptr, dtype=np.int32),
        Bj=np.asarray(B.indices, dtype=np.int32),
        Bx=B.data,
        
        k=np.int32(k),
        lower_bound=lower_bound,
        
        Cj=I,
        Cx=D,
    )
    
    I = np.array(I).reshape(num_rows, k)
    D = np.array(D).reshape(num_rows, k)
    
    return D, I


def run_topdot2(A, B, k, lower_bound=0):
    A = A.tocsr()
    B = B.tocsr()
    
    num_rows = A.shape[0]
    
    I = np.empty(num_rows * k, dtype=np.int32)
    D = np.empty(num_rows * k, dtype=A.dtype)
    
    topdot._topdot2(
        n_row=np.int32(num_rows),
        n_col=np.int32(B.shape[1]),
        
        Ap=np.asarray(A.indptr, dtype=np.int32),
        Aj=np.asarray(A.indices, dtype=np.int32),
        Ax=A.data,
        
        Bp=np.asarray(B.indptr, dtype=np.int32),
        Bj=np.asarray(B.indices, dtype=np.int32),
        Bx=B.data,
        
        k=np.int32(k),
        lower_bound=lower_bound,
        
        Cj=I,
        Cx=D,
    )
    
    I = np.array(I).reshape(num_rows, k)
    D = np.array(D).reshape(num_rows, k)
    
    return D, I

def _run_naive(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return (None, None)
    elif nnz <= ntop:
        return csr_row.data, csr_row.indices
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        return csr_row.indices[arg_idx], csr_row.data[arg_idx]

def run_naive(A, B, ntop, lower_bound=None):
    C = A.dot(B)
    I, D = zip(*[_run_naive(row, ntop) for row in C])
    return D, I

# --
# Run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=4096)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    t = time()
    A = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    B = sparse.rand(args.dim, args.dim, density=args.density, format='csr')
    gen_time = time() - t
    print('gen_time  ', gen_time, file=sys.stderr)
    
    t = time()
    td_D, td_I = run_topdot(A, B, args.k)
    td_time = time() - t
    print('td_time   ', td_time, file=sys.stderr)

    t = time()
    td_D2, td_I2 = run_topdot2(A, B, args.k)
    td_time2 = time() - t
    print('td_time   ', td_time2, file=sys.stderr)
    
    t = time()
    na_D, na_I = run_naive(A, B, args.k)
    naive_time = time() - t
    print('naive_time', naive_time, file=sys.stderr)
    
    rand_idx = np.random.choice(args.dim, args.k, replace=False)
    for idx in rand_idx:
        na_idx = sorted(na_I[idx])
        td_idx = sorted(td_I[idx])
        assert (na_idx == td_idx), "(na_idx != td_idx)"
    
    t = time()
    cc = A.dot(B)
    dot_time = time() - t
    print('dot_time  ', dot_time, file=sys.stderr)
    
    print(json.dumps({
        "gen_time"   : gen_time,
        "td_time"    : td_time,
        "td_time2"   : td_time2,
        "naive_time" : naive_time,
        "dot_time"   : dot_time,
    }))
    